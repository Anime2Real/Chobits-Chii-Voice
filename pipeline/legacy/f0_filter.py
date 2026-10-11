#!/usr/bin/env python3
"""CREPE 音高过滤: 把明显是男声的片段从收录/复核集中移出.

规则: 中位 F0(仅取高置信浊音帧) < THRESHOLD Hz 判为男声.
  dataset/review/ 中的男声 -> dataset/review_male/
  dataset/wavs/   中的男声 -> dataset/wavs_lowpitch/ (不删除, 人工决定), 并从 metadata.csv 移除对应行
F0 明细写 dataset/f0_crepe.csv

用法: .venv/bin/python pipeline/legacy/f0_filter.py
"""
import csv
import glob
import os
from math import gcd

import numpy as np
import soundfile as sf
import torch
import torchcrepe
from scipy.signal import resample_poly

THRESHOLD = 200.0  # Hz, 小叽观测下限约 250, 男性正常说话 100-180
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"


def median_f0(path):
    y, sr = sf.read(path, dtype="float32")
    if sr != 16000:
        g = gcd(sr, 16000)
        y = resample_poly(y, 16000 // g, sr // g)
    a = torch.tensor(y).unsqueeze(0).to(DEVICE)
    f0, conf = torchcrepe.predict(a, 16000, 256, 50, 700, model="full",
                                  device=DEVICE, return_periodicity=True)
    f0 = f0.squeeze().cpu().numpy()
    conf = conf.squeeze().cpu().numpy()
    voiced = f0[conf > 0.5]
    if len(voiced) < 3:
        return None
    return float(np.median(voiced))


def main():
    moved = {"review": [], "wavs": []}
    rows = []
    for group in ("review", "wavs"):
        folder = f"dataset/{group}"
        for path in sorted(glob.glob(os.path.join(folder, "*.wav"))):
            f0 = median_f0(path)
            rows.append({"group": group, "file": os.path.basename(path),
                         "f0": round(f0, 1) if f0 else ""})
            if f0 is not None and f0 < THRESHOLD:
                moved[group].append(os.path.basename(path))

    with open("dataset/f0_crepe.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["group", "file", "f0"])
        w.writeheader()
        w.writerows(rows)

    for group, dest in (("review", "review_male"), ("wavs", "wavs_lowpitch")):
        os.makedirs(f"dataset/{dest}", exist_ok=True)
        for name in moved[group]:
            os.rename(f"dataset/{group}/{name}", f"dataset/{dest}/{name}")
        print(f"{group}: 移出 {len(moved[group])} 条男声 -> dataset/{dest}/")

    if moved["wavs"]:
        drop = {os.path.splitext(n)[0] for n in moved["wavs"]}
        with open("dataset/metadata.csv", encoding="utf-8") as f:
            lines = [l for l in f if l.split("|", 1)[0] not in drop]
        with open("dataset/metadata.csv", "w", encoding="utf-8") as f:
            f.writelines(lines)
        print(f"metadata.csv 移除 {len(drop)} 行, 剩余 {len(lines)} 行")


if __name__ == "__main__":
    main()
