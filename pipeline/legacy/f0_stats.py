#!/usr/bin/env python3
"""F0(基频)分析: 统计给定目录下 wav 的中位基频分布, 用于男女声筛选.

用法: .venv/bin/python pipeline/legacy/f0_stats.py dataset/wavs dataset/review
"""
import csv
import glob
import os
import sys

import librosa
import numpy as np


def median_f0(path):
    y, sr = librosa.load(path, sr=22050, mono=True)
    f0 = librosa.yin(y, fmin=65, fmax=600, sr=sr, frame_length=2048)
    voiced = f0[f0 > 0]
    if len(voiced) < 3:
        return None  # 几乎无浊音(气声/机械音/纯噪声)
    return float(np.median(voiced))


def main():
    out_csv = "dataset/f0_stats.csv"
    rows = []
    for folder in sys.argv[1:]:
        group = os.path.basename(folder.rstrip("/"))
        for path in sorted(glob.glob(os.path.join(folder, "*.wav"))):
            f0 = median_f0(path)
            rows.append({"group": group, "file": os.path.basename(path),
                         "f0": round(f0, 1) if f0 else ""})
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["group", "file", "f0"])
        w.writeheader()
        w.writerows(rows)

    for group in sorted({r["group"] for r in rows}):
        vals = [r["f0"] for r in rows if r["group"] == group and r["f0"]]
        vals = np.array(vals, dtype=float)
        n_novoice = sum(1 for r in rows if r["group"] == group and not r["f0"])
        print(f"[{group}] n={len(vals)} (无浊音 {n_novoice})")
        print("  F0 分位数:", np.percentile(vals, [5, 25, 50, 75, 95]).round(0))
        # 直方图: <165(男声区) / 165-220(过渡) / >220(小叽区)
        lo = (vals < 165).sum()
        mid = ((vals >= 165) & (vals < 220)).sum()
        hi = (vals >= 220).sum()
        print(f"  <165Hz(男声区): {lo}  165-220Hz: {mid}  >=220Hz(女声区): {hi}")


if __name__ == "__main__":
    main()
