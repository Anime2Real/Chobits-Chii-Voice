#!/usr/bin/env python3
"""把人工标注的片段嵌入成向量, 存 annotations/human_labels.npz (files/labels/embs).

A 组从 build/candidates|review 读, B/C 组从 build/labeling/clips 读 (均为 22050Hz),
重采样到 16k 后用 ECAPA 嵌入. 已存在则跳过 (增量: 只补新文件).

用法: .venv/bin/python pipeline/embed_human_labels.py
"""
import json
import os

import numpy as np
import soundfile as sf
import torch
from scipy.signal import resample_poly
from speechbrain.inference.speaker import EncoderClassifier

NPZ_PATH = os.path.join("annotations", "human_labels.npz")
# 与 label_ui.py 的 AUDIO_DIRS 保持一致; "wavs" 兼容历史清单里 src=wavs 的条目
DIRS = {"wavs": os.path.join("dataset", "wavs"),
        "candidates": os.path.join("build", "candidates"),
        "review": os.path.join("build", "review"),
        "clips": os.path.join("build", "labeling", "clips")}


def main():
    clips = {c["file"]: c for c in json.load(open("annotations/clips.json", encoding="utf-8"))}
    labels = json.load(open("annotations/labels.json", encoding="utf-8"))

    old_files, old_labels, old_embs = [], [], np.zeros((0, 192))
    if os.path.exists(NPZ_PATH):
        z = np.load(NPZ_PATH)
        old_files, old_labels, old_embs = z["files"].tolist(), z["labels"].tolist(), z["embs"]
    done = set(old_files)
    todo = [(f, labels[f]) for f in labels if f not in done and f in clips]
    print(f"已缓存 {len(done)} 条, 新增 {len(todo)} 条")

    embs = list(old_embs)
    if todo:
        encoder = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb", run_opts={"device": "cpu"})
        for n, (f, lab) in enumerate(todo):
            c = clips[f]
            wav, sr = sf.read(os.path.join(DIRS[c["src"]], f + ".wav"), dtype="float32")
            clip16 = resample_poly(wav, 16000, sr)  # 22050 -> 16000
            with torch.no_grad():
                e = encoder.encode_batch(
                    torch.tensor(clip16).unsqueeze(0)).squeeze().cpu().numpy()
            embs.append(e / (np.linalg.norm(e) + 1e-8))
            old_files.append(f)
            old_labels.append(lab)
            if (n + 1) % 50 == 0:
                print(f"  {n + 1}/{len(todo)}", flush=True)

    np.savez(NPZ_PATH, files=np.array(old_files), labels=np.array(old_labels),
             embs=np.stack(embs) if embs else np.zeros((0, 192)))
    print(f"保存 {NPZ_PATH}: {len(old_files)} 条")


if __name__ == "__main__":
    main()
