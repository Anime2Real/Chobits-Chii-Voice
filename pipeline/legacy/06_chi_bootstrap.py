#!/usr/bin/env python3
"""小叽参考向量自举 + 相似度打分导出.

思路: 收集文本恰为"ちい/チー"类独白的 segments, 平均其 embedding 得到小叽参考向量,
对全部 segments 算 cosine 相似度, 导出 top-N 试听片段.

用法: .venv/bin/python pipeline/legacy/06_chi_bootstrap.py pilot/ep01/clusters pilot/ep01/vocals_16k.wav pilot/ep01/chi_candidates [topn]
"""
import json
import os
import re
import sys

import numpy as np
import soundfile as sf

CHI_PATTERN = re.compile(r"^[ちチ][ぃいー]{1,2}[!！?？]?$")  # ちい/チー/ちぃ?/チィー! 等整句独白


def main():
    clu_dir, wav_path, out_dir = sys.argv[1], sys.argv[2], sys.argv[3]
    topn = int(sys.argv[4]) if len(sys.argv) > 4 else 20
    os.makedirs(out_dir, exist_ok=True)

    X = np.load(f"{clu_dir}/embeddings.npy")
    with open(f"{clu_dir}/kept_segments.json", encoding="utf-8") as f:
        kept = json.load(f)
    audio, sr = sf.read(wav_path)

    ref_idx = [i for i, s in enumerate(kept) if CHI_PATTERN.match(s["text"])]
    if not ref_idx:
        raise SystemExit("没有匹配到'ちい'类独白, 无法自举参考向量")
    ref = X[ref_idx].mean(axis=0)
    ref /= np.linalg.norm(ref)
    print(f"参考向量来自 {len(ref_idx)} 段: {[kept[i]['text'] for i in ref_idx]}")

    sims = X @ ref
    order = np.argsort(-sims)[:topn]
    results = []
    for rank, i in enumerate(order):
        s = kept[i]
        a = audio[int(s["start"] * sr):int(s["end"] * sr)]
        name = f"{rank:02d}_sim{sims[i]:.3f}_{s['start']:.0f}s.wav"
        sf.write(os.path.join(out_dir, name), a, sr)
        results.append({"rank": rank, "sim": round(float(sims[i]), 3), **s})
        print(f"{rank:3d}  sim={sims[i]:.3f}  {s['start']:8.2f}-{s['end']:8.2f}  {s['text']}")
    with open(os.path.join(out_dir, "candidates.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
