#!/usr/bin/env python3
"""聚类阈值扫描: 观察不同 cosine 距离阈值下的聚类粒度.

用法: .venv/bin/python pipeline/legacy/05_sweep_threshold.py pilot/ep01/clusters
"""
import json
import sys

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import pdist

out_dir = sys.argv[1]
X = np.load(f"{out_dir}/embeddings.npy")
with open(f"{out_dir}/kept_segments.json", encoding="utf-8") as f:
    kept = json.load(f)

Z = linkage(pdist(X, metric="cosine"), method="average")
durs = np.array([s["end"] - s["start"] for s in kept])

print(f"{'thr':>5} {'n_clu':>6} {'top1_seg':>9} {'top1_dur':>9} {'top3_dur':>9}")
for thr in [0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95, 1.1, 1.25]:
    labels = fcluster(Z, t=thr, criterion="distance")
    sizes = {}
    dur_sums = {}
    for lab, d in zip(labels, durs):
        sizes[lab] = sizes.get(lab, 0) + 1
        dur_sums[lab] = dur_sums.get(lab, 0) + d
    top_durs = sorted(dur_sums.values(), reverse=True)[:3]
    top_lab = max(sizes, key=sizes.get)
    print(f"{thr:>5} {len(sizes):>6} {sizes[top_lab]:>9} "
          f"{max(dur_sums.values()):>8.0f}s {sum(top_durs):>8.0f}s")
