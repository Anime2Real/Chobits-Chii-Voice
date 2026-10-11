#!/usr/bin/env python3
"""用人工确认的候选片段构建小叽参考向量, 并用它重新打分验证分离度.

用法: .venv/bin/python pipeline/legacy/07_make_reference.py pilot/ep01/clusters pilot/ep01/chi_candidates/candidates.json pilot/chi_reference.npy 0 1 2 3 4 6 7 8
"""
import json
import sys

import numpy as np

clu_dir, cand_path, out_path = sys.argv[1], sys.argv[2], sys.argv[3]
ranks = [int(x) for x in sys.argv[4:]]

X = np.load(f"{clu_dir}/embeddings.npy")
kept = json.load(open(f"{clu_dir}/kept_segments.json", encoding="utf-8"))
cands = json.load(open(cand_path, encoding="utf-8"))

start2idx = {round(s["start"], 2): i for i, s in enumerate(kept)}
ref_idx = [start2idx[round(cands[r]["start"], 2)] for r in ranks]
ref = X[ref_idx].mean(axis=0)
ref /= np.linalg.norm(ref)
np.save(out_path, ref)
print(f"参考向量: {len(ref_idx)} 段 -> {out_path}")

# 用新参考向量重打分, 看确认/否定样本的分离情况
sims = X @ ref
order = np.argsort(-sims)[:25]
for rank, i in enumerate(order):
    s = kept[i]
    print(f"{rank:3d}  sim={sims[i]:.3f}  {s['start']:8.2f}-{s['end']:8.2f}  {s['text']}")
