#!/usr/bin/env python3
"""第六轮导出: 扩充负样本池(77 条) + 两段式筛选.

与 round5 的差异:
  1. 负样本池 = 75 条人工标注 + 2 条混音标注 = 77 条
  2. 最短时长 2s -> 1s (回收"ちい"类标志性短句; 纯度交给声纹判别)
  3. 两段式: 第一轮照旧打分收录; 第二轮对 >=4s 的收录片段做能量谷重切,
     子片段重新打分 + 重新转写文本(whisper 的文本属于整段, 切开必须重转写),
     只保留二次过线的子片段 —— 专治"一个窗口里小叽+其他角色"的混入

打分: sim = 与 chi_reference_v4 相似度; margin = sim - 负样本池最近邻相似度
收录: sim >= T1 且 margin >= MARGIN

用法: .venv/bin/python pipeline/legacy/round6.py
"""
import csv
import json
import os
import shutil

import mlx_whisper
import numpy as np
import soundfile as sf
import torch
from scipy.signal import resample_poly
from speechbrain.inference.speaker import EncoderClassifier

import batch
from round4 import cut_trim, is_quiet, split_long

T1, MARGIN = 0.55, -0.05
MIN_DUR, MAX_DUR = 1.0, 10.0
RESPLIT_DUR = 4.0                 # 超过此时长的收录片段进入第二轮重切
WHISPER_REPO = "mlx-community/whisper-large-v3-turbo"
OUT_DIR = batch.OUT_DIR


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


def main():
    chi = np.load("pilot/chi_reference_v4.npy")
    lab = np.load("pilot/labeled_embeddings.npz")
    neg_E = np.concatenate([lab["negative"], lab["mixed"]])

    encoder = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb", run_opts={"device": "cpu"})

    def embed(clip16):
        with torch.no_grad():
            e = encoder.encode_batch(
                torch.tensor(clip16, dtype=torch.float32).unsqueeze(0)).squeeze().cpu().numpy()
        return e / (np.linalg.norm(e) + 1e-8)

    def score(clip16):
        emb = embed(clip16)
        sc = float(emb @ chi)
        return sc, sc - float((neg_E @ emb).max())

    for sub in ("wavs", "review"):
        shutil.rmtree(os.path.join(OUT_DIR, sub), ignore_errors=True)
        os.makedirs(os.path.join(OUT_DIR, sub))

    rows, full_rows = [], []
    clip_no, n_resplit, n_resplit_kept = 0, 0, 0
    for label, _ in batch.find_episodes():
        if "." in label:
            continue
        ep_dir = os.path.join(batch.BUILD_DIR, batch.ep_dir_name(label))
        kp = os.path.join(ep_dir, "kept.json")
        if not os.path.exists(kp):
            continue
        with open(kp, encoding="utf-8") as f:
            kept = json.load(f)
        audio16k, sr16 = sf.read(os.path.join(ep_dir, "vocals_16k.wav"))
        va, sr_v = sf.read(os.path.join(ep_dir, "separated", "htdemucs", "audio", "vocals.wav"),
                           dtype="float32")

        # 第一轮: 候选 chunk(长句词级拆分) + 打分
        chunks = []
        for seg in kept:
            dur = seg["end"] - seg["start"]
            if dur < MIN_DUR:
                continue
            if dur <= MAX_DUR:
                chunks.append(seg)
            else:
                chunks.extend(split_long(seg, audio16k, sr16))

        n_chi = 0
        for seg in chunks:
            s, e = int(seg["start"] * sr16), int(seg["end"] * sr16)
            clip16 = audio16k[s:e]
            if is_quiet(clip16):
                continue
            sc, mg = score(clip16)
            if not (sc >= T1 and mg >= MARGIN):
                continue

            # 第二轮: >=4s 的收录片段能量谷重切 + 子片段重打分
            pieces = [(seg["start"], seg["end"])]
            if seg["end"] - seg["start"] >= RESPLIT_DUR:
                pieces = valley_split(audio16k, sr16, seg["start"], seg["end"])
                if len(pieces) > 1:
                    n_resplit += 1
            for ps, pe in pieces:
                if pe - ps < MIN_DUR:
                    continue
                pclip = audio16k[int(ps * sr16):int(pe * sr16)]
                if is_quiet(pclip):
                    continue
                if len(pieces) > 1:
                    sc2, mg2 = score(pclip)
                    if not (sc2 >= T1 and mg2 >= MARGIN):
                        continue
                    # 切开必须重转写: 整段文本不属于子片段
                    r = mlx_whisper.transcribe(
                        pclip.astype("float32"), path_or_hf_repo=WHISPER_REPO,
                        language="ja", condition_on_previous_text=False)
                    text = "".join(x["text"] for x in r["segments"]).strip() or seg["text"]
                else:
                    text = seg["text"]
                n_resplit_kept += 1
                clip = resample_poly(cut_trim(va, sr_v, ps, pe), 1, 2)
                if is_quiet(clip):
                    continue
                full_rows.append({"ep": label, "start": round(ps, 2), "end": round(pe, 2),
                                  "sim": round(sc, 3), "margin": round(mg, 3), "text": text})
                clip_no += 1
                name = f"{clip_no:06d}"
                sf.write(os.path.join(OUT_DIR, "wavs", f"{name}.wav"), clip, 22050,
                         subtype="PCM_16")
                rows.append((name, text))
                n_chi += 1
        print(f"  第{label}话 chunks={len(chunks)} exported={n_chi}", flush=True)

    with open(os.path.join(OUT_DIR, "metadata.csv"), "w", encoding="utf-8") as f:
        for name, text in rows:
            f.write(f"{name}|{text}\n")
    with open(os.path.join(OUT_DIR, "metadata_full.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["ep", "start", "end", "sim", "margin", "text"])
        w.writeheader()
        w.writerows(full_rows)
    print(f"完成: {clip_no} 段收录; 第二轮重切 {n_resplit} 段, 子片段保留 {n_resplit_kept} 条")
    print("(本轮无 review 带: 未过线片段已由两轮筛除, 边界样本见 metadata_full.csv 低分行)")


if __name__ == "__main__":
    main()
