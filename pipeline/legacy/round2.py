#!/usr/bin/env python3
"""第二轮: 用第一轮高置信片段重建小叽参考向量, 重打分并按双档阈值重新导出.

阶段:
  1. 补齐总集篇(8.5/16.5/24.5)的音轨/分离/转写(复用 batch.py, 幂等)
  2. 为全部剧集计算并缓存 embedding (build/epXX/embeddings.npy + kept.json)
  3. 参考向量 v2 = 第一轮 sim>=REF_SIM 且时长>=REF_DUR 的片段 embedding 均值
  4. 全量重打分, >=T1 收录进 dataset/, [T2,T1) 进 dataset/review/

用法: .venv/bin/python pipeline/legacy/round2.py
"""
import csv
import json
import os
import shutil

import numpy as np
import soundfile as sf
import torch
from scipy.signal import resample_poly
from speechbrain.inference.speaker import EncoderClassifier

import batch

REF_SIM = 0.70   # 第一轮高置信线, 用于挑选参考样本
REF_DUR = 2.0    # 参考样本最短时长(秒), 保证 embedding 质量
T1 = 0.65        # 自动收录阈值
T2 = 0.55        # 复核带下界
MIN_DUR, MAX_DUR = batch.MIN_DUR, batch.MAX_DUR
OUT_DIR = batch.OUT_DIR


def embed_episode(encoder, ep_dir):
    """计算并缓存一集的 embedding, 幂等. 返回 (kept, X) 或 (None, None)."""
    emb_path = os.path.join(ep_dir, "embeddings.npy")
    kept_path = os.path.join(ep_dir, "kept.json")
    if os.path.exists(emb_path) and os.path.exists(kept_path):
        with open(kept_path, encoding="utf-8") as f:
            kept = json.load(f)
        return kept, np.load(emb_path)
    seg_path = os.path.join(ep_dir, "segments.json")
    if not os.path.exists(seg_path):
        return None, None
    audio16k, sr = sf.read(os.path.join(ep_dir, "vocals_16k.wav"))
    with open(seg_path, encoding="utf-8") as f:
        segments = json.load(f)
    kept, embs = [], []
    for seg in segments:
        s, e = int(seg["start"] * sr), int(seg["end"] * sr)
        if (e - s) / sr < 0.5:
            continue
        wav = torch.tensor(audio16k[s:e], dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            emb = encoder.encode_batch(wav).squeeze().cpu().numpy()
        embs.append(emb / (np.linalg.norm(emb) + 1e-8))
        kept.append(seg)
    if not embs:
        return None, None
    X = np.stack(embs)
    np.save(emb_path, X)
    with open(kept_path, "w", encoding="utf-8") as f:
        json.dump(kept, f, ensure_ascii=False, indent=1)
    return kept, X


def main():
    encoder = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb", run_opts={"device": "cpu"})

    # 阶段 1+2: 补齐总集篇, 缓存全部 embedding
    ep_data = {}
    for label, mp4_path in batch.find_episodes():
        ep_dir = os.path.join(batch.BUILD_DIR, batch.ep_dir_name(label))
        print(f"===== 第{label}话 =====", flush=True)
        vocals, vocals16k = batch.prepare_episode(label, mp4_path, ep_dir)
        batch.transcribe(vocals16k, os.path.join(ep_dir, "segments.json"),
                         batch.op_ed_ranges(label))
        kept, X = embed_episode(encoder, ep_dir)
        if kept is not None:
            ep_data[label] = {"dir": ep_dir, "vocals": vocals, "kept": kept, "X": X}
        print(f"  segments={0 if kept is None else len(kept)}", flush=True)

    # 阶段 3: 用第一轮高置信片段构建参考向量 v2
    ref_embs = []
    with open(os.path.join(OUT_DIR, "metadata_full.csv"), encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if float(row["sim"]) < REF_SIM:
                continue
            if float(row["end"]) - float(row["start"]) < REF_DUR:
                continue
            label = row["ep"]
            if label not in ep_data:
                continue
            kept, X = ep_data[label]["kept"], ep_data[label]["X"]
            for i, s in enumerate(kept):
                if abs(s["start"] - float(row["start"])) < 0.01:
                    ref_embs.append(X[i])
                    break
    if not ref_embs:
        raise SystemExit("没有挑到参考样本, 检查 REF_SIM/REF_DUR")
    ref = np.stack(ref_embs).mean(axis=0)
    ref /= np.linalg.norm(ref)
    np.save("pilot/chi_reference_v2.npy", ref)
    print(f"参考向量 v2: {len(ref_embs)} 段 -> pilot/chi_reference_v2.npy")

    # 阶段 4: 重打分 + 重新导出
    for sub in ("wavs", "review"):
        shutil.rmtree(os.path.join(OUT_DIR, sub), ignore_errors=True)
        os.makedirs(os.path.join(OUT_DIR, sub))
    rows, full_rows = [], []
    clip_no = 0
    for label, d in ep_data.items():
        sims = d["X"] @ ref
        vocals_audio, sr_v = sf.read(d["vocals"], dtype="float32")
        n_chi = 0
        for seg, sim in zip(d["kept"], sims):
            dur = seg["end"] - seg["start"]
            if not (MIN_DUR <= dur <= MAX_DUR) or sim < T2:
                continue
            s, e = int(seg["start"] * sr_v), int(seg["end"] * sr_v)
            clip = resample_poly(vocals_audio[s:e].mean(axis=1), 1, 2)
            # 静音检查: whisper 在预告/转场弱人声间隙会幻听出文本
            rms_db = 20 * np.log10(float(np.sqrt((clip ** 2).mean())) + 1e-12)
            if rms_db < -55 or float(np.abs(clip).max()) < 0.025:
                continue
            full_rows.append({"ep": label, "start": seg["start"], "end": seg["end"],
                              "sim": round(float(sim), 3), "text": seg["text"]})
            if sim < T1:
                sf.write(os.path.join(OUT_DIR, "review",
                                      f"{batch.ep_dir_name(label)}_sim{sim:.3f}_{seg['start']:.0f}s.wav"),
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
        w = csv.DictWriter(f, fieldnames=["ep", "start", "end", "sim", "text"])
        w.writeheader()
        w.writerows(full_rows)
    sims_all = np.array([r["sim"] for r in full_rows])
    print(f"完成: {clip_no} 段收录, {len(full_rows) - clip_no} 段待复核")
    print("候选 sim 分位数:", np.percentile(sims_all, [10, 25, 50, 75, 90]).round(3))


if __name__ == "__main__":
    main()
