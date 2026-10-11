#!/usr/bin/env python3
"""第八轮导出: 人工标注训练的逻辑回归分类器打分.

与 round7 的差异:
  1. 打分器 = pilot/chi_lr.pkl (embedding [+F0 特征] -> 小叽概率),
     阈值由 597 条人工标注交叉验证选定 (目标: 高精确率)
  2. >10s 且词级拆分失败的长句: 能量谷兜底切分, 不再直接丢弃
  3. F0 特征 (CREPE 中位数/四分位/周期性) 按集缓存, 仅当模型需要时计算

用法: .venv/bin/python pipeline/legacy/round8.py
"""
import csv
import json
import os
import pickle
import shutil

import mlx_whisper
import numpy as np
import soundfile as sf
import torch
from scipy.signal import resample_poly
from speechbrain.inference.speaker import EncoderClassifier

import batch
from common import get_chunks, get_embeddings
from round4 import cut_trim, is_quiet
from round6 import valley_split

MIN_DUR, MAX_DUR = 1.0, 10.0
RESPLIT_DUR = 4.0
WHISPER_REPO = "mlx-community/whisper-large-v3-turbo"
OUT_DIR = batch.OUT_DIR


def clip_name(label, start):
    return f"{batch.ep_dir_name(label)}_{start:08.2f}s"


def f0_of(clip16, device):
    """CREPE 基频特征: [中位数, p25, p75, 平均周期性]. 无声帧返回全 0."""
    import torchcrepe
    y = torch.tensor(clip16, dtype=torch.float32).unsqueeze(0).to(device)
    f0, per = torchcrepe.predict(y, 16000, 80, 80, 800, model="tiny",
                                 device=device, return_periodicity=True)
    f0, per = f0.squeeze().cpu().numpy(), per.squeeze().cpu().numpy()
    v = f0[per > 0.5]
    if len(v) < 3:
        return [0.0, 0.0, 0.0, 0.0]
    return [float(np.median(v)), float(np.percentile(v, 25)),
            float(np.percentile(v, 75)), float(per.mean())]


def get_f0(ep_dir, idx, chunks, audio16k, sr, device):
    """按 chunk embedding 缓存对齐的 F0 特征缓存."""
    cache = os.path.join(ep_dir, f"chunk_f0_{MIN_DUR}_{MAX_DUR}.npy")
    if os.path.exists(cache):
        return np.load(cache)
    F = np.array([f0_of(audio16k[int(chunks[i]["start"] * sr):int(chunks[i]["end"] * sr)],
                        device) for i in idx])
    np.save(cache, F)
    return F


def main():
    with open("pilot/chi_lr.pkl", "rb") as f:
        scorer = pickle.load(f)
    clf, thr, use_f0 = scorer["model"], scorer["threshold"], scorer.get("use_f0", False)
    print(f"分类器阈值 {thr}, 使用 F0 特征: {use_f0}")
    f0_device = "mps" if use_f0 and torch.backends.mps.is_available() else "cpu"

    encoder = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb", run_opts={"device": "cpu"})

    def embed(clip16):
        with torch.no_grad():
            e = encoder.encode_batch(
                torch.tensor(clip16, dtype=torch.float32).unsqueeze(0)).squeeze().cpu().numpy()
        return e / (np.linalg.norm(e) + 1e-8)

    def prob(clip16, emb=None):
        e = embed(clip16) if emb is None else emb
        feats = np.concatenate([e, f0_of(clip16, f0_device)]) if use_f0 else e
        return float(clf.predict_proba(feats.reshape(1, -1))[0, 1])

    for sub in ("wavs", "review"):
        shutil.rmtree(os.path.join(OUT_DIR, sub), ignore_errors=True)
        os.makedirs(os.path.join(OUT_DIR, sub))

    rows, full_rows = [], []
    n_resplit = n_fallback = 0
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

        chunks = get_chunks(ep_dir, kept, audio16k, sr16, MIN_DUR, MAX_DUR)
        idx, X = get_embeddings(encoder, ep_dir, chunks, audio16k, sr16, MIN_DUR, MAX_DUR)
        feats = np.concatenate([X, get_f0(ep_dir, idx, chunks, audio16k, sr16, f0_device)],
                               axis=1) if use_f0 else X
        probs = clf.predict_proba(feats)[:, 1] if len(feats) else np.array([])

        # 词级拆分失败的 >10s 长句: 能量谷兜底, 切成 <=10s 伪 chunk
        covered = [(chunks[i]["start"], chunks[i]["end"]) for i in idx]
        for seg in kept:
            if seg["end"] - seg["start"] <= MAX_DUR:
                continue
            if any(s >= seg["start"] - 0.01 and e <= seg["end"] + 0.01
                   for s, e in covered):
                continue
            pieces = valley_split(audio16k, sr16, seg["start"], seg["end"])
            fixed = []
            for s, e in pieces:
                while e - s > MAX_DUR:
                    fixed.append((s, s + MAX_DUR))
                    s += MAX_DUR
                fixed.append((s, e))
            for s, e in fixed:
                e = min(e, len(audio16k) / sr16)  # whisper 时间戳可能越过音频末尾
                if e - s >= MIN_DUR:
                    chunks.append({"start": s, "end": e, "text": seg["text"]})
                    idx.append(len(chunks) - 1)
                    p = prob(audio16k[int(s * sr16):int(e * sr16)])
                    probs = np.append(probs, p)
                    n_fallback += 1

        n_chi = 0
        for seg, p in zip((chunks[i] for i in idx), probs):
            if p < thr:
                continue
            pieces = [(seg["start"], seg["end"])]
            if seg["end"] - seg["start"] >= RESPLIT_DUR:
                sub = valley_split(audio16k, sr16, seg["start"], seg["end"])
                if len(sub) > 1:
                    pieces = sub
                    n_resplit += 1
            for ps, pe in pieces:
                if pe - ps < MIN_DUR:
                    continue
                pclip = audio16k[int(ps * sr16):int(pe * sr16)]
                if is_quiet(pclip):
                    continue
                if len(pieces) > 1:
                    p2 = prob(pclip)
                    if p2 < thr:
                        continue
                    r = mlx_whisper.transcribe(
                        pclip.astype("float32"), path_or_hf_repo=WHISPER_REPO,
                        language="ja", condition_on_previous_text=False)
                    text = "".join(x["text"] for x in r["segments"]).strip() or seg["text"]
                else:
                    p2 = p
                    text = seg["text"]
                clip = resample_poly(cut_trim(va, sr_v, ps, pe), 1, 2)
                if is_quiet(clip):
                    continue
                name = clip_name(label, ps)
                sf.write(os.path.join(OUT_DIR, "wavs", f"{name}.wav"), clip, 22050,
                         subtype="PCM_16")
                rows.append((name, text))
                full_rows.append({"ep": label, "start": round(ps, 2), "end": round(pe, 2),
                                  "prob": round(p2, 3), "text": text, "file": name})
                n_chi += 1
        print(f"  第{label}话 chunks={len(chunks)} exported={n_chi}", flush=True)

    with open(os.path.join(OUT_DIR, "metadata.csv"), "w", encoding="utf-8") as f:
        for name, text in rows:
            f.write(f"{name}|{text}\n")
    with open(os.path.join(OUT_DIR, "metadata_full.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["file", "ep", "start", "end", "prob", "text"])
        w.writeheader()
        w.writerows(full_rows)
    print(f"完成: {len(rows)} 段收录; 重切 {n_resplit} 段; 长句兜底 {n_fallback} 段")


if __name__ == "__main__":
    main()
