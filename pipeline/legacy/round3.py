#!/usr/bin/env python3
"""第三轮: 纯净参考向量 + 判别式打分重新导出.

打分(基于 cached embedding):
  sim_v2   = 与 v2 参考(69 段完整句)的相似度 —— 保证整句召回
  margin   = sim_v2 - max(sim_male, sim_female) —— 男女声参考由人工标注的 39 条错例构建
收录: 时长 2~10s 且非静音 且 sim_v2 >= T1 且 margin >= MARGIN
复核: 未收录但 sim_v2 >= T2 且 margin >= MARGIN_REVIEW

另导出 dataset/labeling/ : 30 条跨剧集分层抽样候选, 供人工确认后构建 v4 参考.

用法: .venv/bin/python pipeline/legacy/round3.py
"""
import csv
import json
import os
import shutil

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

import batch

T1, MARGIN = 0.62, -0.02          # 收录线
T2, MARGIN_REVIEW = 0.55, -0.08   # 复核带
MIN_DUR, MAX_DUR = 2.0, 10.0
OUT_DIR = batch.OUT_DIR


def main():
    chi = np.load("pilot/chi_reference_v2.npy")
    male = np.load("pilot/male_reference.npy")
    female = np.load("pilot/female_reference.npy")

    for sub in ("wavs", "review", "labeling"):
        shutil.rmtree(os.path.join(OUT_DIR, sub), ignore_errors=True)
        os.makedirs(os.path.join(OUT_DIR, sub))

    rows, full_rows = [], []
    clip_no = 0
    pool = []  # (label, seg, sc, mg) 供 labeling 抽样
    for label, _ in batch.find_episodes():
        ep_dir = os.path.join(batch.BUILD_DIR, batch.ep_dir_name(label))
        kp, xp = os.path.join(ep_dir, "kept.json"), os.path.join(ep_dir, "embeddings.npy")
        if not (os.path.exists(kp) and os.path.exists(xp)):
            continue
        with open(kp, encoding="utf-8") as f:
            kept = json.load(f)
        X = np.load(xp)
        sims = X @ chi
        margins = sims - np.maximum(X @ male, X @ female)

        vocals = os.path.join(ep_dir, "separated", "htdemucs", "audio", "vocals.wav")
        vocals_audio, sr_v = sf.read(vocals, dtype="float32")

        def cut(seg):
            s, e = int(seg["start"] * sr_v), int(seg["end"] * sr_v)
            return resample_poly(vocals_audio[s:e].mean(axis=1), 1, 2)

        n_chi = 0
        for seg, sc, mg in zip(kept, sims, margins):
            dur = seg["end"] - seg["start"]
            if not (MIN_DUR <= dur <= MAX_DUR):
                continue
            accepted = sc >= T1 and mg >= MARGIN
            in_review = (not accepted) and sc >= T2 and mg >= MARGIN_REVIEW
            if not (accepted or in_review):
                continue
            clip = cut(seg)
            rms_db = 20 * np.log10(float(np.sqrt((clip ** 2).mean())) + 1e-12)
            if rms_db < -55 or float(np.abs(clip).max()) < 0.025:
                continue
            full_rows.append({"ep": label, "start": seg["start"], "end": seg["end"],
                              "sim": round(float(sc), 3), "margin": round(float(mg), 3),
                              "text": seg["text"]})
            pool.append({"ep": label, "seg": seg, "sc": float(sc), "mg": float(mg),
                         "clip": clip})
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
        print(f"  第{label}话 exported={n_chi}", flush=True)

    with open(os.path.join(OUT_DIR, "metadata.csv"), "w", encoding="utf-8") as f:
        for name, text in rows:
            f.write(f"{name}|{text}\n")
    with open(os.path.join(OUT_DIR, "metadata_full.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["ep", "start", "end", "sim", "margin", "text"])
        w.writeheader()
        w.writerows(full_rows)
    print(f"完成: {clip_no} 段收录, {len(full_rows) - clip_no} 段待复核")

    # labeling: 跨剧集分层, 优先 3-8s 完整句, 高置信与边界各半
    cands = [p for p in pool if p["mg"] >= 0]
    cands.sort(key=lambda p: -p["mg"])
    by_ep = {}
    for p in cands:
        by_ep.setdefault(p["ep"], []).append(p)
    picked = []
    for ep, items in by_ep.items():  # 每集最高分 1 条
        picked.append(items[0])
    boundary = sorted(cands, key=lambda p: abs(p["mg"] - 0.06))
    for p in boundary:  # 边界样本补足到 30
        if len(picked) >= 30:
            break
        if p not in picked:
            picked.append(p)
    for i, p in enumerate(picked[:30]):
        sf.write(os.path.join(OUT_DIR, "labeling",
                              f"{i:02d}_{batch.ep_dir_name(p['ep'])}_sim{p['sc']:.2f}_m{p['mg']:+.2f}_{p['seg']['start']:.0f}s.wav"),
                 p["clip"], 22050, subtype="PCM_16")
    print(f"labeling: {min(30, len(picked))} 条 -> dataset/labeling/")


if __name__ == "__main__":
    main()
