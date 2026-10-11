#!/usr/bin/env python3
"""说话人聚类: 对每个 segment 提取 ECAPA-TDNN embedding, 凝聚聚类.

用法: .venv/bin/python pipeline/legacy/04_cluster.py pilot/ep01/vocals_16k.wav pilot/ep01/segments.json pilot/ep01/clusters
输出: clusters/assignments.json, clusters/cluster_<id>/sample_<i>.wav (每类最多 5 个试听样本)
"""
import json
import os
import sys

import numpy as np
import soundfile as sf
import torch
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import pdist
from speechbrain.inference.speaker import EncoderClassifier

MIN_SEG = 0.5  # 秒, 过短片段不做 embedding
DIST_THRESHOLD = 0.35  # cosine 距离阈值, 越小越严格


def main():
    wav_path, seg_path, out_dir = sys.argv[1], sys.argv[2], sys.argv[3]
    os.makedirs(out_dir, exist_ok=True)

    audio, sr = sf.read(wav_path)
    assert sr == 16000
    with open(seg_path, encoding="utf-8") as f:
        segments = json.load(f)

    # speechbrain 1.1 不支持 mps device_type, embedding 用 CPU 即可
    device = "cpu"
    encoder = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        run_opts={"device": device},
    )

    kept, embeddings = [], []
    for seg in segments:
        s, e = int(seg["start"] * sr), int(seg["end"] * sr)
        if (e - s) / sr < MIN_SEG:
            continue
        wav = torch.tensor(audio[s:e], dtype=torch.float32).unsqueeze(0).to(device)
        with torch.no_grad():
            emb = encoder.encode_batch(wav).squeeze().cpu().numpy()
        emb /= np.linalg.norm(emb) + 1e-8
        kept.append(seg)
        embeddings.append(emb)

    X = np.stack(embeddings)
    np.save(os.path.join(out_dir, "embeddings.npy"), X)
    with open(os.path.join(out_dir, "kept_segments.json"), "w", encoding="utf-8") as f:
        json.dump(kept, f, ensure_ascii=False, indent=1)
    Z = linkage(pdist(X, metric="cosine"), method="average")
    labels = fcluster(Z, t=DIST_THRESHOLD, criterion="distance")

    assignments = []
    for seg, lab in zip(kept, labels):
        assignments.append({**seg, "cluster": int(lab)})
    with open(os.path.join(out_dir, "assignments.json"), "w", encoding="utf-8") as f:
        json.dump(assignments, f, ensure_ascii=False, indent=1)

    # 每类导出最多 5 个试听样本
    counts = {}
    for a, emb in zip(assignments, embeddings):
        counts.setdefault(a["cluster"], []).append(a)
    summary = []
    for cid, items in sorted(counts.items(), key=lambda kv: -len(kv[1])):
        total = sum(i["end"] - i["start"] for i in items)
        summary.append({"cluster": cid, "n_segments": len(items), "total_sec": round(total, 1)})
        cdir = os.path.join(out_dir, f"cluster_{cid:02d}")
        os.makedirs(cdir, exist_ok=True)
        # 取时长最长的 5 段作为试听样本
        for i, item in enumerate(sorted(items, key=lambda x: -(x["end"] - x["start"]))[:5]):
            s, e = int(item["start"] * sr), int(item["end"] * sr)
            sf.write(os.path.join(cdir, f"sample_{i}_{item['start']:.0f}s.wav"), audio[s:e], sr)
        with open(os.path.join(cdir, "texts.txt"), "w", encoding="utf-8") as f:
            for item in items:
                f.write(f"{item['start']:8.2f}  {item['text']}\n")
    print(json.dumps(summary, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
