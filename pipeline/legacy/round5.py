#!/usr/bin/env python3
"""第五轮导出: v4 参考(8 短句 + 24 条人工确认长句) + 最近邻负样本判别.

与 round4 的机制相同(长句词级拆分/边界能量谷修剪/跳总集篇/静音前移),
差异在打分:
  sim      = 与 chi_reference_v4 的余弦相似度
  nn_neg   = 与 43 条人工标注负样本的最大相似度(最近邻)
  margin   = sim - nn_neg
收录: 2-10s 且非静音 且 sim >= T1 且 margin >= MARGIN
混音嫌疑: 满足收录但 nn_neg >= NN_MIX 的进 review(文件名带 mix 标记)
复核带: sim >= T2 且 margin >= MARGIN_REVIEW

chunk 与 embedding 按集缓存(build/epXX/chunks.json / chunk_embeddings.npy), 重跑快.

用法: .venv/bin/python pipeline/legacy/round5.py
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

T1, MARGIN = 0.60, -0.02          # 收录线(负样本 margin 最大 -0.157, 留有 buffer)
NN_MIX = 0.75                     # 混音嫌疑线(标注混音 nn 为 0.61/0.71; 正常小叽 nn 也常达 0.6+, 阈值取高)
T2, MARGIN_REVIEW = 0.55, -0.05   # 复核带
MIN_DUR, MAX_DUR = 2.0, 10.0
OUT_DIR = batch.OUT_DIR


def get_chunks(ep_dir, kept, audio16k, sr):
    """按集缓存的候选 chunk 列表(长句已拆分)."""
    cache = os.path.join(ep_dir, "chunks.json")
    if os.path.exists(cache):
        with open(cache, encoding="utf-8") as f:
            return json.load(f)
    chunks = []
    for seg in kept:
        dur = seg["end"] - seg["start"]
        if dur < MIN_DUR:
            continue
        if dur <= MAX_DUR:
            chunks.append(seg)
        else:
            chunks.extend(split_long(seg, audio16k, sr))
    with open(cache, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=1)
    return chunks


def get_embeddings(encoder, ep_dir, chunks, audio16k, sr):
    """按集缓存的 chunk embedding; 静音 chunk 不缓存不返回."""
    cache = os.path.join(ep_dir, "chunk_embeddings.npy")
    idx_cache = os.path.join(ep_dir, "chunk_index.json")
    if os.path.exists(cache) and os.path.exists(idx_cache):
        with open(idx_cache, encoding="utf-8") as f:
            idx = json.load(f)
        return idx, np.load(cache)
    idx, embs = [], []
    for i, seg in enumerate(chunks):
        s, e = int(seg["start"] * sr), int(seg["end"] * sr)
        clip = audio16k[s:e]
        if is_quiet(clip):
            continue
        with torch.no_grad():
            emb = encoder.encode_batch(
                torch.tensor(clip, dtype=torch.float32).unsqueeze(0)).squeeze().cpu().numpy()
        idx.append(i)
        embs.append(emb / (np.linalg.norm(emb) + 1e-8))
    X = np.stack(embs) if embs else np.zeros((0, 192))
    np.save(cache, X)
    with open(idx_cache, "w", encoding="utf-8") as f:
        json.dump(idx, f)
    return idx, X


def main():
    chi = np.load("pilot/chi_reference_v4.npy")
    neg_E = np.load("pilot/labeled_embeddings.npz")["negative"]

    encoder = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb", run_opts={"device": "cpu"})

    for sub in ("wavs", "review"):
        shutil.rmtree(os.path.join(OUT_DIR, sub), ignore_errors=True)
        os.makedirs(os.path.join(OUT_DIR, sub))

    rows, full_rows = [], []
    clip_no = 0
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
        va, sr_v = sf.read(os.path.join(ep_dir, "separated", "htdemucs", "audio", "vocals.wav"),
                           dtype="float32")

        chunks = get_chunks(ep_dir, kept, audio16k, sr16)
        idx, X = get_embeddings(encoder, ep_dir, chunks, audio16k, sr16)
        sims = X @ chi
        nns = (X @ neg_E.T).max(axis=1) if len(X) else np.array([])
        margins = sims - nns

        n_chi = 0
        for seg, sc, nn, mg in zip((chunks[i] for i in idx), sims, nns, margins):
            accepted = sc >= T1 and mg >= MARGIN
            mixed = accepted and nn >= NN_MIX
            in_review = (not accepted) and sc >= T2 and mg >= MARGIN_REVIEW
            if mixed:
                accepted, in_review = False, True
            if not (accepted or in_review):
                continue
            clip = resample_poly(cut_trim(va, sr_v, seg["start"], seg["end"]), 1, 2)
            if is_quiet(clip):
                continue
            full_rows.append({"ep": label, "start": seg["start"], "end": seg["end"],
                              "sim": round(float(sc), 3), "margin": round(float(mg), 3),
                              "nn_neg": round(float(nn), 3), "text": seg["text"]})
            if in_review:
                tag = "mix_" if mixed else ""
                sf.write(os.path.join(OUT_DIR, "review",
                                      f"{tag}{batch.ep_dir_name(label)}_sim{sc:.3f}_m{mg:+.3f}_{seg['start']:.0f}s.wav"),
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
        w = csv.DictWriter(f, fieldnames=["ep", "start", "end", "sim", "margin", "nn_neg", "text"])
        w.writeheader()
        w.writerows(full_rows)
    print(f"完成: {clip_no} 段收录, {len(full_rows) - clip_no} 段待复核")


if __name__ == "__main__":
    main()
