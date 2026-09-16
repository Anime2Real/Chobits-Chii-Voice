#!/usr/bin/env python3
"""生成全集台词索引 build/transcripts.csv (由 finalize.py 发布到 dataset/).

每行一句 whisper 转写台词 (OP/ED 已剔除), 附带:
  chi_prob   句内所有 chunk 的小叽分类概率最大值 (可粗略判断是谁说的)
  in_dataset 该句是否有片段入选最终数据集 (build/metadata_full.csv)

用途: 全文检索"是否说过某句话", 并定位到集数和时间.
  例: grep 'ちい' dataset/transcripts.csv

用法: .venv/bin/python pipeline/build_transcript_index.py
"""
import csv
import io
import json
import os
import pickle

import numpy as np
import soundfile as sf
from speechbrain.inference.speaker import EncoderClassifier

import batch
import metadata_schema as ms
from common import get_chunks, get_embeddings

MIN_DUR, MAX_DUR = 1.0, 10.0


def main():
    with open("annotations/chi_lr.pkl", "rb") as f:
        clf = pickle.load(f)["model"]
    encoder = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb", run_opts={"device": "cpu"})

    # 已收录片段区间, 用于 in_dataset 标记
    meta_path = os.path.join(batch.BUILD_DIR, "metadata_full.csv")
    if not os.path.exists(meta_path):
        raise SystemExit(f"找不到 {meta_path}, 请先运行: .venv/bin/python pipeline/batch.py")
    exported = {}
    for r in ms.read_metadata_full(meta_path):
        exported.setdefault(r["ep"], []).append((r["start"], r["end"]))

    rows = []
    for label, _ in batch.find_episodes():
        if "." in label:
            continue
        ep_name = batch.ep_dir_name(label)
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

        for seg in kept:
            s, e = seg["start"], seg["end"]
            p = max((pr for i, pr in zip(idx, probs)
                     if chunks[i]["start"] < e and chunks[i]["end"] > s), default=None)
            in_ds = any(s < xe and e > xs
                        for xs, xe in exported.get(ep_name, []))
            rows.append({"ep": ep_name, "start": round(s, 2), "end": round(e, 2),
                         "dur": round(e - s, 2), "text": seg["text"],
                         "chi_prob": round(float(p), 3) if p is not None else "",
                         "in_dataset": int(in_ds)})
        print(f"  第{label}话 {len(kept)} 句", flush=True)

    out = os.path.join(batch.BUILD_DIR, "transcripts.csv")
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=["ep", "start", "end", "dur", "text",
                                        "chi_prob", "in_dataset"], lineterminator="\r\n")
    w.writeheader()
    w.writerows(rows)
    ms.atomic_write_text(out, buf.getvalue())
    n_in = sum(r["in_dataset"] for r in rows)
    print(f"完成: {len(rows)} 句 -> {out}; 其中 {n_in} 句有片段入选数据集")
    print("finalize.py 会把它发布到 dataset/transcripts.csv")


if __name__ == "__main__":
    main()
