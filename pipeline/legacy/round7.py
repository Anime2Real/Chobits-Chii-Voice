#!/usr/bin/env python3
"""第七轮导出: 规则同 round6, 但产物稳定化.

  1. chunk / embedding 全部按集缓存(pipeline/common.py), 重跑只需秒级
  2. 导出文件名改为内容寻址: ep05_754.30s.wav (集数_起始秒), 跨轮次稳定,
     人工标注的文件名不会因重导出而失效
  3. 打分: sim = 与 chi_reference_v4 相似度; margin = sim - 77 条负样本最近邻
  4. 两段式: >=4s 收录片段能量谷重切, 子片段重打分 + 重转写

用法: .venv/bin/python pipeline/legacy/round7.py
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
from common import get_chunks, get_embeddings
from round4 import cut_trim, is_quiet
from round6 import valley_split

T1, MARGIN = 0.55, -0.05
MIN_DUR, MAX_DUR = 1.0, 10.0
RESPLIT_DUR = 4.0
WHISPER_REPO = "mlx-community/whisper-large-v3-turbo"
OUT_DIR = batch.OUT_DIR


def clip_name(label, start):
    return f"{batch.ep_dir_name(label)}_{start:08.2f}s"


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

    for sub in ("wavs", "review"):
        shutil.rmtree(os.path.join(OUT_DIR, sub), ignore_errors=True)
        os.makedirs(os.path.join(OUT_DIR, sub))

    rows, full_rows = [], []
    n_resplit = 0
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
        sims = X @ chi
        margins = sims - (X @ neg_E.T).max(axis=1) if len(X) else np.array([])

        n_chi = 0
        for seg, sc, mg in zip((chunks[i] for i in idx), sims, margins):
            if not (sc >= T1 and mg >= MARGIN):
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
                    emb = embed(pclip)
                    sc2 = float(emb @ chi)
                    mg2 = sc2 - float((neg_E @ emb).max())
                    if not (sc2 >= T1 and mg2 >= MARGIN):
                        continue
                    r = mlx_whisper.transcribe(
                        pclip.astype("float32"), path_or_hf_repo=WHISPER_REPO,
                        language="ja", condition_on_previous_text=False)
                    text = "".join(x["text"] for x in r["segments"]).strip() or seg["text"]
                else:
                    text = seg["text"]
                clip = resample_poly(cut_trim(va, sr_v, ps, pe), 1, 2)
                if is_quiet(clip):
                    continue
                name = clip_name(label, ps)
                sf.write(os.path.join(OUT_DIR, "wavs", f"{name}.wav"), clip, 22050,
                         subtype="PCM_16")
                rows.append((name, text))
                full_rows.append({"ep": label, "start": round(ps, 2), "end": round(pe, 2),
                                  "sim": round(float(sc), 3), "margin": round(float(mg), 3),
                                  "text": text, "file": name})
                n_chi += 1
        print(f"  第{label}话 chunks={len(chunks)} exported={n_chi}", flush=True)

    with open(os.path.join(OUT_DIR, "metadata.csv"), "w", encoding="utf-8") as f:
        for name, text in rows:
            f.write(f"{name}|{text}\n")
    with open(os.path.join(OUT_DIR, "metadata_full.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["file", "ep", "start", "end", "sim", "margin", "text"])
        w.writeheader()
        w.writerows(full_rows)
    print(f"完成: {len(rows)} 段收录; 重切 {n_resplit} 段")


if __name__ == "__main__":
    main()
