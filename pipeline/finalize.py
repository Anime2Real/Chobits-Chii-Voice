#!/usr/bin/env python3
"""最终导出: 人工确认的小叽片段 + 高置信自动片段 -> dataset/.

数据来源:
  1. annotations/labels.json 中标记为 chi 的片段
     (音频在 build/candidates, build/review 和 build/labeling/clips,
     重转写文本); 同集重叠 >50% 的去重, 保留更长者
  2. 未被人工标注、LR 概率 >= THR_AUTO 的新 chunk (>=4s 能量谷重切 + 重打分 + 重转写),
     与人工确认片段重叠 >50% 的跳过

依赖 batch.py 在 build/ 下的中间产物 (kept.json / 分轨 / metadata_full.csv).

产出 (dataset/ 下唯一的写入入口, 全部原子写出):
  dataset/wavs/             22050Hz 16bit 单声道, 内容寻址命名
  dataset/metadata.csv      file|text
  dataset/metadata_full.csv file,ep,start,end,prob,source,text
  dataset/transcripts.csv   若 build/transcripts.csv 存在则一并发布

用法: .venv/bin/python pipeline/finalize.py
"""
import json
import os
import pickle
import shutil

import mlx_whisper
import numpy as np
import soundfile as sf
import torch
from scipy.signal import resample_poly
from speechbrain.inference.speaker import EncoderClassifier

import batch
import metadata_schema as ms
from audio_utils import cut_trim, is_quiet, valley_split
from common import get_chunks, get_embeddings

MIN_DUR, MAX_DUR = 1.0, 10.0
RESPLIT_DUR = 4.0
THR_AUTO = 0.96
WHISPER_REPO = "mlx-community/whisper-large-v3-turbo"
OUT_DIR = batch.OUT_DIR
LABELS_PATH = os.path.join("annotations", "labels.json")
# 人工片段音频的可能位置, 按优先级探测 ("candidates"/"review" 由 batch.py 产出,
# "clips" 由 prepare_labeling.py 产出, "wavs" 兼容历史清单里 src=wavs 的条目)
SRC_DIRS = {"candidates": os.path.join(batch.BUILD_DIR, "candidates"),
            "review": os.path.join(batch.BUILD_DIR, "review"),
            "clips": os.path.join(batch.BUILD_DIR, "labeling", "clips"),
            "wavs": os.path.join("dataset", "wavs")}

ep_dir_name = ms.ep_dir_name
clip_name = ms.clip_name


def overlap(a_start, a_end, b_start, b_end):
    inter = max(0.0, min(a_end, b_end) - max(a_start, b_start))
    return inter / max(a_end - a_start, 1e-6) > 0.5


def find_src(file):
    """按优先级探测人工片段的音频位置, 找不到返回 None (走 build/ 重建)."""
    for src, d in SRC_DIRS.items():
        if os.path.exists(os.path.join(d, file + ".wav")):
            return src
    return None


def transcribe(clip16):
    r = mlx_whisper.transcribe(clip16.astype("float32"), path_or_hf_repo=WHISPER_REPO,
                               language="ja", condition_on_previous_text=False)
    return "".join(x["text"] for x in r["segments"]).strip()


def main():
    if not os.path.exists(LABELS_PATH):
        raise SystemExit(f"找不到 {LABELS_PATH}, 请先运行 label_ui.py 完成人工标注")
    with open(LABELS_PATH, encoding="utf-8") as f:
        labels = json.load(f)

    with open("annotations/chi_lr.pkl", "rb") as f:
        clf = pickle.load(f)["model"]
    encoder = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb", run_opts={"device": "cpu"})

    def embed(clip16):
        with torch.no_grad():
            e = encoder.encode_batch(
                torch.tensor(clip16, dtype=torch.float32).unsqueeze(0)).squeeze().cpu().numpy()
        return e / (np.linalg.norm(e) + 1e-8)

    # 先解析人工片段的音频位置, 再清空输出目录
    # (否则 wavs/ 里旧的同名文件会被自己删掉, 只能靠 build/ 重建)
    src_of = {f: find_src(f) for f, lab in labels.items() if lab == "chi"}
    shutil.rmtree(os.path.join(OUT_DIR, "wavs"), ignore_errors=True)
    os.makedirs(os.path.join(OUT_DIR, "wavs"))

    # ---- 1. 人工确认片段: 解析文件名得到 (ep, start), 去重 ----
    human = {}  # ep -> [(start, end, file, src)]
    for f, lab in labels.items():
        if lab != "chi":
            continue
        src = src_of[f]
        ep_part, t_part = f.rsplit("_", 1)
        start = float(t_part[:-1])
        if src is not None:
            wav, sr = sf.read(os.path.join(SRC_DIRS[src], f + ".wav"))
            human.setdefault(ep_part, []).append((start, start + len(wav) / sr, f, src))
        else:
            human.setdefault(ep_part, []).append((start, start + 3.0, f, None))  # 待重建

    ep_cache = {}

    def rebuild(ep_name, start):
        """从 build/ 重建缺失片段: 找包含 start 的 chunk, 能量谷重切取起点最近的子段."""
        if ep_name not in ep_cache:
            ep_dir = os.path.join(batch.BUILD_DIR, ep_name)
            with open(os.path.join(ep_dir, "chunks_1.0_10.0.json"), encoding="utf-8") as f:
                chunks = json.load(f)
            a16, s16 = sf.read(os.path.join(ep_dir, "vocals_16k.wav"))
            v, sv = sf.read(os.path.join(ep_dir, "separated", "htdemucs", "audio", "vocals.wav"),
                            dtype="float32")
            ep_cache[ep_name] = (chunks, a16, s16, v, sv)
        chunks, a16, s16, v, sv = ep_cache[ep_name]
        c = min(chunks, key=lambda c: abs(c["start"] - start))
        pieces = [(c["start"], c["end"])]
        if c["end"] - c["start"] >= RESPLIT_DUR:
            sub = valley_split(a16, s16, c["start"], c["end"])
            if len(sub) > 1:
                pieces = sub
        ps, pe = min(pieces, key=lambda p: abs(p[0] - start))
        pe = min(pe, len(a16) / s16)
        clip = resample_poly(cut_trim(v, sv, ps, pe), 1, 2)
        return clip, ps, pe

    kept_rows, kept_intervals = [], {}
    for ep, items in sorted(human.items()):
        items.sort(key=lambda x: -(x[1] - x[0]))  # 长的优先保留
        for s, e, f, src in items:
            if any(overlap(s, e, xs, xe) for xs, xe in kept_intervals.get(ep, [])):
                continue
            kept_intervals.setdefault(ep, []).append((s, e))
            kept_rows.append((ep, s, e, f, src))

    rows, full_rows = [], []
    n_human = 0
    for ep, s, e, f, src in sorted(kept_rows):
        if src is not None:
            wav, sr = sf.read(os.path.join(SRC_DIRS[src], f + ".wav"), dtype="float32")
        else:
            wav, ps, pe = rebuild(ep, s)  # 音频不在暂存区时, 从 build/ 重建
            sr = 22050
            s, e = ps, pe
            kept_intervals.setdefault(ep, []).append((ps, pe))
        clip16 = resample_poly(wav, 16000, sr)
        text = transcribe(clip16)
        sf.write(os.path.join(OUT_DIR, "wavs", f + ".wav"), wav, sr, subtype="PCM_16")
        rows.append((f, text))
        full_rows.append(ms.make_row(file=f, ep=ep, start=s, end=e, prob="",
                                     source=ms.SOURCE_HUMAN, text=text))
        n_human += 1
        if n_human % 100 == 0:
            print(f"  人工片段转写 {n_human}/{len(kept_rows)}", flush=True)

    # ---- 2. 高置信自动片段 ----
    n_auto = 0
    for label, _ in batch.find_episodes():
        if "." in label:
            continue
        ep_dir = os.path.join(batch.BUILD_DIR, ep_dir_name(label))
        kp = os.path.join(ep_dir, "kept.json")
        if not os.path.exists(kp):
            continue
        with open(kp, encoding="utf-8") as f:
            kept = json.load(f)
        audio16k, sr16 = sf.read(os.path.join(ep_dir, "vocals_16k.wav"))
        va, sr_v = sf.read(os.path.join(ep_dir, "separated", "htdemucs", "audio", "vocals.wav"),
                           dtype="float32")
        chunks = get_chunks(ep_dir, kept, audio16k, sr16, MIN_DUR, MAX_DUR)
        idx, X = get_embeddings(encoder, ep_dir, chunks, audio16k, sr16, MIN_DUR, MAX_DUR)
        probs = clf.predict_proba(X)[:, 1] if len(X) else np.array([])
        ep_name = ep_dir_name(label)

        for seg, p in zip((chunks[i] for i in idx), probs):
            if p < THR_AUTO or clip_name(ep_name, seg["start"]) in labels:
                continue
            pieces = [(seg["start"], seg["end"])]
            if seg["end"] - seg["start"] >= RESPLIT_DUR:
                sub = valley_split(audio16k, sr16, seg["start"], seg["end"])
                if len(sub) > 1:
                    pieces = sub
            for ps, pe in pieces:
                pe = min(pe, len(audio16k) / sr16)
                if pe - ps < MIN_DUR:
                    continue
                if any(overlap(ps, pe, xs, xe) for xs, xe in kept_intervals.get(ep_name, [])):
                    continue
                pclip = audio16k[int(ps * sr16):int(pe * sr16)]
                if is_quiet(pclip):
                    continue
                if len(pieces) > 1:
                    if float(clf.predict_proba(embed(pclip).reshape(1, -1))[0, 1]) < THR_AUTO:
                        continue
                    text = transcribe(pclip) or seg["text"]
                else:
                    text = seg["text"]
                clip = resample_poly(cut_trim(va, sr_v, ps, pe), 1, 2)
                if is_quiet(clip):
                    continue
                name = clip_name(ep_name, ps)
                sf.write(os.path.join(OUT_DIR, "wavs", f"{name}.wav"), clip, 22050,
                         subtype="PCM_16")
                rows.append((name, text))
                full_rows.append(ms.make_row(file=name, ep=ep_name, start=ps, end=pe,
                                             prob=p, source=ms.SOURCE_AUTO, text=text))
                kept_intervals.setdefault(ep_name, []).append((ps, pe))
                n_auto += 1
        print(f"  第{label}话 自动收录累计={n_auto}", flush=True)

    ms.write_metadata(os.path.join(OUT_DIR, "metadata.csv"), rows)
    ms.write_metadata_full(os.path.join(OUT_DIR, "metadata_full.csv"), full_rows)

    # 台词索引由 build_transcript_index.py 生成到 build/, 此处一并发布
    idx_src = os.path.join(batch.BUILD_DIR, "transcripts.csv")
    idx_dst = os.path.join(OUT_DIR, "transcripts.csv")
    if os.path.exists(idx_src):
        with open(idx_src, encoding="utf-8") as f:
            ms.atomic_write_text(idx_dst, f.read())
    else:
        print("提示: 未找到 build/transcripts.csv, 跳过发布"
              "(可先运行 .venv/bin/python pipeline/build_transcript_index.py)")

    total = sum(r["end"] - r["start"] for r in full_rows)
    print(f"\n完成: 人工确认 {n_human} 段 + 自动 {n_auto} 段 = {len(rows)} 段"
          f" / {total / 60:.1f} 分钟")


if __name__ == "__main__":
    main()
