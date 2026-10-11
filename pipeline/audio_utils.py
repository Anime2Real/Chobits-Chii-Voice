#!/usr/bin/env python3
"""音频切片工具函数: 静音判定 / 长句词级拆分 / 能量谷切分与修剪.

历史来源: 原 round4.py (is_quiet/split_long/cut_trim) 与 round6.py (valley_split).
"""
import mlx_whisper
import numpy as np

MIN_DUR, MAX_DUR = 1.0, 10.0
PAD, SEARCH = 0.15, 0.20  # 导出扩边 / 能量谷搜索范围(秒)
WHISPER_REPO = "mlx-community/whisper-large-v3-turbo"


def is_quiet(y):
    """静音/近静音判定 (16k 切片, embedding 前调用)."""
    if len(y) == 0:
        return True
    rms_db = 20 * np.log10(float(np.sqrt((y ** 2).mean())) + 1e-12)
    return rms_db < -55 or float(np.abs(y).max()) < 0.025


def split_long(seg, audio16k, sr):
    """用词级时间戳把 >10s 的段切成 1-10s 小块, 返回绝对时间的 chunk 列表."""
    s0, e0 = int(seg["start"] * sr), int(seg["end"] * sr)
    window = audio16k[s0:e0].astype("float32")
    result = mlx_whisper.transcribe(
        window, path_or_hf_repo=WHISPER_REPO, language="ja",
        word_timestamps=True, condition_on_previous_text=False)
    words = [w for s in result["segments"] for w in s.get("words", [])]
    if not words:
        return []
    chunks, cur, cur_start = [], [], words[0]["start"]
    for w in words:
        gap = w["start"] - (cur[-1]["end"] if cur else w["start"])
        cur_dur = w["end"] - cur_start
        if cur and (cur_dur > MAX_DUR or (gap >= 0.4 and (cur[-1]["end"] - cur_start) >= MIN_DUR)):
            chunks.append(cur)
            cur, cur_start = [], w["start"]
        cur.append(w)
    if cur:
        chunks.append(cur)
    out = []
    for ws in chunks:
        s, e = seg["start"] + ws[0]["start"], seg["start"] + ws[-1]["end"]
        if e - s >= MIN_DUR:
            out.append({"start": s, "end": e,
                        "text": "".join(w["word"] for w in ws).strip()})
    return out


def valley_split(audio, sr, start, end, _depth=0):
    """在 [start,end] 内按能量谷(静音点)切分, 返回绝对时间 (s,e) 列表.

    递归切超长片段限制 _depth 层: 谷在边界附近时递归收敛极慢(尾部折半),
    超过深度直接返回超长片段, 由调用方定长切分兜底."""
    s0, e0 = int(start * sr), int(end * sr)
    clip = np.abs(audio[s0:e0])
    fl = int(0.02 * sr)
    n = len(clip) // fl
    if n < 3:
        return [(start, end)]
    env = clip[: n * fl].reshape(n, fl).max(axis=1)
    thr = 0.12 * env.max()
    quiet = env < thr
    # 连续静音 >=0.25s 记为一个切点
    cuts = []
    i = 0
    while i < n:
        if quiet[i]:
            j = i
            while j < n and quiet[j]:
                j += 1
            if (j - i) * 0.02 >= 0.25:
                cuts.append((i + j) / 2 * 0.02)
            i = j
        else:
            i += 1
    bounds = [0.0] + cuts + [end - start]
    pieces = []
    for a, b in zip(bounds, bounds[1:]):
        if b - a >= MIN_DUR:
            pieces.append((start + a, start + b))
    # 超长片段在最安静的谷继续切
    out = []
    for a, b in pieces:
        if b - a <= MAX_DUR or _depth >= 3:
            out.append((a, b))
        else:
            sub = valley_split(audio, sr, a, b, _depth + 1)
            out.extend(sub if len(sub) > 1 else [(a, b)])
    return out or [(start, end)]


def cut_trim(va, sr, start, end):
    """扩边后按能量谷修剪, 返回 44.1k 单声道切片."""
    s0 = max(0.0, start - PAD)
    e0 = min(len(va) / sr, end + PAD)
    clip = va[int(s0 * sr):int(e0 * sr)].mean(axis=1)
    fl = int(0.01 * sr)
    n = len(clip) // fl
    if n < 3:
        return clip
    e = (clip[: n * fl].reshape(n, fl) ** 2).mean(axis=1)
    w = int(SEARCH / 0.01)
    i0 = int(np.argmin(e[: w + 1]))
    j0 = n - 1 - int(np.argmin(e[::-1][: w + 1]))
    if j0 <= i0:
        return clip
    return clip[i0 * fl:(j0 + 1) * fl]
