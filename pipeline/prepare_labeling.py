#!/usr/bin/env python3
"""准备人工标注数据: 用 LR 分类器概率挑选待标注片段, 生成标注清单.

三组数据 (均跳过 labels.json 中已有标签的片段):
  A = build/candidates|review 已收录的片段 (复核纯度)
  B = prob 在 [B_LO, 收录阈值) 的边界片段 (分类器最纠结的区域)
  C = prob 在 [C_LO, B_LO) 的随机抽查 (验证更低分区是否漏掉小叽)

读取:
  build/metadata_full.csv      batch.py 产出的候选+复核总表 (metadata_schema 格式)
  build/epXX/kept.json         句段缓存 (batch.py 产出)

产出 (均为可再生中间产物):
  build/labeling/clips/        B/C 组片段 wav (22050Hz, 命名内容寻址)
  annotations/clips.json       标注清单 (label_ui.py 读取)

用法: .venv/bin/python pipeline/prepare_labeling.py
"""
import json
import os
import pickle
import random

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly
from speechbrain.inference.speaker import EncoderClassifier

import batch
import metadata_schema as ms
from common import get_chunks, get_embeddings
from audio_utils import cut_trim, is_quiet

MIN_DUR, MAX_DUR = 1.0, 10.0
B_LO = 0.45                     # B 组: prob 在 [B_LO, 收录阈值)
C_LO, C_PER_EP = 0.25, 2        # C 组: prob 在 [C_LO, B_LO), 每集抽查数
LAB_DIR = os.path.join(batch.BUILD_DIR, "labeling")
CLIPS_DIR = os.path.join(LAB_DIR, "clips")

ep_dir_name = ms.ep_dir_name
clip_name = ms.clip_name


def overlap(a_start, a_end, b_start, b_end):
    inter = max(0.0, min(a_end, b_end) - max(a_start, b_start))
    return inter / max(a_end - a_start, 1e-6) > 0.5


def main():
    os.makedirs(CLIPS_DIR, exist_ok=True)

    # A 组: 已收录候选/复核片段, 从 build/metadata_full.csv 读分数
    # 已标注的文件不再出现在清单中 (标签已是定论)
    labels_path = os.path.join("annotations", "labels.json")
    labeled = set()
    if os.path.exists(labels_path):
        with open(labels_path, encoding="utf-8") as f:
            labeled = set(json.load(f))

    with open("annotations/chi_lr.pkl", "rb") as f:
        scorer = pickle.load(f)
    clf, thr = scorer["model"], scorer["threshold"]

    meta_path = os.path.join(batch.BUILD_DIR, "metadata_full.csv")
    if not os.path.exists(meta_path):
        raise SystemExit(f"找不到 {meta_path}, 请先运行: .venv/bin/python pipeline/batch.py")
    exported, group_a = {}, []
    for r in ms.read_metadata_full(meta_path):
        exported.setdefault(r["ep"], []).append((r["start"], r["end"]))
        if r["file"] in labeled:
            continue
        group_a.append({
            "file": r["file"], "src": "candidates" if r["source"] == ms.SOURCE_CANDIDATE else "review",
            "group": "A", "ep": r["ep"], "start": r["start"], "end": r["end"],
            "dur": round(r["end"] - r["start"], 2),
            "prob": r["prob"], "text": r["text"],
        })

    encoder = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb", run_opts={"device": "cpu"})
    rng = random.Random(42)
    group_b, group_c = [], []

    for label, _ in batch.find_episodes():
        if "." in label:
            continue
        ep_name = ep_dir_name(label)
        ep_dir = os.path.join(batch.BUILD_DIR, ep_name)
        kp = os.path.join(ep_dir, "kept.json")
        if not os.path.exists(kp):
            continue
        with open(kp, encoding="utf-8") as f:
            kept = json.load(f)
        audio16k, sr16 = sf.read(os.path.join(ep_dir, "vocals_16k.wav"))

        chunks = get_chunks(ep_dir, kept, audio16k, sr16, MIN_DUR, MAX_DUR)
        idx, X = get_embeddings(encoder, ep_dir, chunks, audio16k, sr16, MIN_DUR, MAX_DUR)
        probs = clf.predict_proba(X)[:, 1] if len(X) else np.array([])

        b_cand, c_cand = [], []
        for seg, p in zip((chunks[i] for i in idx), probs):
            if p >= thr:
                continue  # 已收录 (A 组)
            s, e = seg["start"], seg["end"]
            if any(overlap(s, e, xs, xe) for xs, xe in exported.get(ep_name, [])):
                continue  # 与收录片段重叠 (重切子段), 不重复标注
            if clip_name(ep_name, s) in labeled:
                continue  # 已有定论
            entry = {"seg": seg, "prob": float(p)}
            if p >= B_LO:
                b_cand.append(entry)
            elif p >= C_LO:
                c_cand.append(entry)
        b_cand.sort(key=lambda x: -x["prob"])  # 离收录线最近的排前面
        picked = b_cand + rng.sample(c_cand, min(C_PER_EP, len(c_cand)))

        va, sr_v = sf.read(os.path.join(ep_dir, "separated", "htdemucs", "audio", "vocals.wav"),
                           dtype="float32")
        for p in picked:
            seg = p["seg"]
            clip = resample_poly(cut_trim(va, sr_v, seg["start"], seg["end"]), 1, 2)
            if is_quiet(clip):
                continue
            name = clip_name(ep_name, seg["start"])
            sf.write(os.path.join(CLIPS_DIR, f"{name}.wav"), clip, 22050, subtype="PCM_16")
            entry = {"file": name, "src": "clips",
                     "group": "B" if p["prob"] >= B_LO else "C",
                     "ep": label, "start": round(seg["start"], 2), "end": round(seg["end"], 2),
                     "dur": round(seg["end"] - seg["start"], 2),
                     "prob": round(p["prob"], 3), "text": seg["text"]}
            (group_b if entry["group"] == "B" else group_c).append(entry)
        print(f"  第{label}话 边界候选={len(b_cand)} 低分候选={len(c_cand)}", flush=True)

    clips = sorted(group_a + group_b + group_c, key=lambda x: (x["ep"], x["start"]))
    clips_path = os.path.join("annotations", "clips.json")
    ms.atomic_write_text(clips_path, json.dumps(clips, ensure_ascii=False, indent=1))
    total_dur = sum(c["dur"] for c in clips)
    print(f"\nA(复核)={len(group_a)}  B(边界)={len(group_b)}  C(抽查)={len(group_c)}"
          f"  共 {len(clips)} 段 / {total_dur / 60:.0f} 分钟音频")
    print(f"清单: {clips_path}")
    print("下一步: .venv/bin/python pipeline/label_ui.py")


if __name__ == "__main__":
    main()
