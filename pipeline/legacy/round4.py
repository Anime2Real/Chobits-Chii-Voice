#!/usr/bin/env python3
"""第四轮导出. 相对第三轮的改动:
  1. >10s 长句用 whisper 词级时间戳在停顿处二次切分到 2-10s, 不再整段丢弃
  2. 跳过总集篇 (8.5/16.5/24.5, 内容与正片重复)
  3. 导出切片 ±150ms 扩边后按能量谷修剪, 避免切到字头字尾
  4. 静音检查前移到 embedding 之前
  5. 导出前先用 47 条人工标注样本校准打分规则

打分: sim = 与 v2 参考的余弦相似度(整句召回); margin = sim - max(sim_male, sim_female)(判别)
收录: 2-10s 且 sim >= T1 且 margin >= MARGIN; 复核带: sim >= T2 且 margin >= MARGIN_REVIEW

注意: v4 参考向量留待 dataset/labeling/ 人工标注完成后构建(见 README/会话), 本轮沿用 v2+判别.

用法: .venv/bin/python pipeline/legacy/round4.py
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

T1, MARGIN = 0.62, -0.02          # 收录线
T2, MARGIN_REVIEW = 0.55, -0.08   # 复核带
MIN_DUR, MAX_DUR = 2.0, 10.0
PAD, SEARCH = 0.15, 0.20          # 导出扩边 / 能量谷搜索范围(秒)
WHISPER_REPO = "mlx-community/whisper-large-v3-turbo"
OUT_DIR = batch.OUT_DIR


def is_quiet(y):
    """静音/近静音判定 (16k 切片, embedding 前调用)."""
    if len(y) == 0:
        return True
    rms_db = 20 * np.log10(float(np.sqrt((y ** 2).mean())) + 1e-12)
    return rms_db < -55 or float(np.abs(y).max()) < 0.025


def split_long(seg, audio16k, sr):
    """用词级时间戳把 >10s 的段切成 2-10s 小块, 返回绝对时间的 chunk 列表."""
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
    for ch in chunks:
        dur = ch[-1]["end"] - ch[0]["start"]
        if dur < MIN_DUR:
            continue
        out.append({"start": seg["start"] + ch[0]["start"], "end": seg["start"] + ch[-1]["end"],
                    "text": "".join(w["word"] for w in ch).strip()})
    return out


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


def main():
    chi = np.load("pilot/chi_reference_v2.npy")
    male = np.load("pilot/male_reference.npy")
    female = np.load("pilot/female_reference.npy")

    # 校准: 打分规则必须拦下全部人工标注负例
    lab = np.load("pilot/labeled_embeddings.npz")
    for group in ("male", "female"):
        E = lab[group]
        sc = E @ chi
        mg = sc - np.maximum(E @ male, E @ female)
        leaked = int(((sc >= T1) & (mg >= MARGIN)).sum())
        print(f"校准[{group}]: {leaked}/{len(E)} 条漏过规则 (应为 0)")
    E = lab["chi"]
    sc = E @ chi
    mg = sc - np.maximum(E @ male, E @ female)
    print(f"校准[chi]: {int(((sc >= T1) & (mg >= MARGIN)).sum())}/{len(E)} 条正例通过")

    encoder = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb", run_opts={"device": "cpu"})

    for sub in ("wavs", "review"):
        shutil.rmtree(os.path.join(OUT_DIR, sub), ignore_errors=True)
        os.makedirs(os.path.join(OUT_DIR, sub))

    rows, full_rows = [], []
    clip_no, n_split, n_split_skip = 0, 0, 0
    for label, _ in batch.find_episodes():
        if "." in label:
            continue  # 跳过总集篇
        ep_dir = os.path.join(batch.BUILD_DIR, batch.ep_dir_name(label))
        kp = os.path.join(ep_dir, "kept.json")
        if not os.path.exists(kp):
            continue
        with open(kp, encoding="utf-8") as f:
            kept = json.load(f)
        audio16k, sr16 = sf.read(os.path.join(ep_dir, "vocals_16k.wav"))
        vocals = os.path.join(ep_dir, "separated", "htdemucs", "audio", "vocals.wav")
        va, sr_v = sf.read(vocals, dtype="float32")

        # 汇总本集候选 chunk (长句拆分)
        chunks = []
        for seg in kept:
            dur = seg["end"] - seg["start"]
            if dur < MIN_DUR:
                continue
            if dur <= MAX_DUR:
                chunks.append(seg)
            else:
                sub = split_long(seg, audio16k, sr16)
                n_split += 1
                n_split_skip += (not sub)
                chunks.extend(sub)

        n_chi = 0
        for seg in chunks:
            s, e = int(seg["start"] * sr16), int(seg["end"] * sr16)
            clip16 = audio16k[s:e]
            if is_quiet(clip16):
                continue
            with torch.no_grad():
                emb = encoder.encode_batch(
                    torch.tensor(clip16, dtype=torch.float32).unsqueeze(0)).squeeze().cpu().numpy()
            emb /= np.linalg.norm(emb) + 1e-8
            sc = float(emb @ chi)
            mg = sc - float(max(emb @ male, emb @ female))
            accepted = sc >= T1 and mg >= MARGIN
            in_review = (not accepted) and sc >= T2 and mg >= MARGIN_REVIEW
            if not (accepted or in_review):
                continue
            clip = resample_poly(cut_trim(va, sr_v, seg["start"], seg["end"]), 1, 2)
            if is_quiet(clip):
                continue
            full_rows.append({"ep": label, "start": seg["start"], "end": seg["end"],
                              "sim": round(sc, 3), "margin": round(mg, 3), "text": seg["text"]})
            if in_review:
                sf.write(os.path.join(OUT_DIR, "review",
                                      f"{batch.ep_dir_name(label)}_sim{sc:.3f}_m{mg:+.3f}_{seg['start']:.0f}s.wav"),
                         clip, 22050, subtype="PCM_16")
                continue
            clip_no += 1
            name = f"{clip_no:06d}"
            sf.write(os.path.join(OUT_DIR, "wavs", f"{name}.wav"), clip, 22050, subtype="PCM_16")
            rows.append((name, seg["text"]))
            n_chi += 1
        print(f"  第{label}话 chunks={len(chunks)} exported={n_chi}", flush=True)

    with open(os.path.join(OUT_DIR, "metadata.csv"), "w", encoding="utf-8") as f:
        for name, text in rows:
            f.write(f"{name}|{text}\n")
    with open(os.path.join(OUT_DIR, "metadata_full.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["ep", "start", "end", "sim", "margin", "text"])
        w.writeheader()
        w.writerows(full_rows)
    print(f"完成: {clip_no} 段收录, {len(full_rows) - clip_no} 段待复核; "
          f"长句拆分 {n_split} 段 (其中 {n_split_skip} 段拆分失败被跳过)")


if __name__ == "__main__":
    main()
