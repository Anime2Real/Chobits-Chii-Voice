#!/usr/bin/env python3
"""批量提取全部剧集的小叽语音候选.

流程(每集): 抽音轨 -> demucs 人声分离 -> mlx-whisper 转写(裁掉 OP/ED)
-> ECAPA embedding -> 与小叽参考向量的 cosine 相似度 -> 阈值分流.

产物 (全部为 build/ 下可再生的中间产物, 本脚本不写 dataset/):
  build/epXX/                     audio.wav / separated/ / vocals_16k.wav (幂等缓存)
  build/epXX/segments.json        whisper 转写 (已剔除 OP/ED)
  build/epXX/embeddings.npy       句级说话人嵌入矩阵 (行与 kept.json 对齐, 幂等缓存)
  build/epXX/kept.json            参与嵌入的句段 [{start,end,text}] (>=0.5s)
  build/epXX/candidates.csv       该集候选+复核片段总表 (metadata_schema 格式)
  build/candidates/epXX_....wav   sim >= THRESHOLD 的候选片段 (22050Hz 单声道 16-bit)
  build/review/epXX_....wav       sim 在 [REVIEW_LO, THRESHOLD) 的待人工复核片段
  build/metadata_full.csv         全部候选+复核行 (source=candidate/review)
  build/review.csv                复核带清单 (source=review 的行)

候选片段按内容寻址命名 (ep05_00667.42s.wav), 跨轮次稳定, 便于溯源.
dataset/ 下的正式文件只由 finalize.py 产出.

用法: .venv/bin/python pipeline/batch.py
环境: CHII_VOICE_DEVICE 覆盖 demucs 设备 (默认 mps); ffmpeg 经 imageio-ffmpeg
解析, 找不到时回退 PATH 中的 ffmpeg.
"""
import json
import os
import re
import shutil
import subprocess

import mlx_whisper
import numpy as np
import soundfile as sf
import torch
from scipy.signal import resample_poly
from speechbrain.inference.speaker import EncoderClassifier

import metadata_schema as ms

MOVIE_DIR = "Chobits_Movie"
BUILD_DIR = "build"
OUT_DIR = "dataset"  # 历史兼容 (legacy/round2.py 引用); batch 本身只写 BUILD_DIR
REF_PATH = "annotations/legacy/chi_reference.npy"
THRESHOLD = 0.60
REVIEW_LO = 0.50
MIN_DUR, MAX_DUR = 1.0, 15.0
CHI_PATTERN = re.compile(r"^[ちチ][ぃいー]{1,2}[!！?？]?$")
N_REF_SAMPLES = 8  # chi_reference.npy 由 8 段平均而成
DEVICE = os.environ.get("CHII_VOICE_DEVICE", "mps")

ep_dir_name = ms.ep_dir_name
clip_name = ms.clip_name

_FFMPEG = None


def ffmpeg_exe():
    """惰性解析 ffmpeg 路径 (首次调用时): 优先 imageio-ffmpeg, 回退 PATH."""
    global _FFMPEG
    if _FFMPEG is None:
        exe = None
        try:
            import imageio_ffmpeg
            exe = imageio_ffmpeg.get_ffmpeg_exe()
        except ImportError:
            exe = shutil.which("ffmpeg")
        if not exe:
            raise RuntimeError(
                "找不到 ffmpeg: 请 pip install imageio-ffmpeg 或将 ffmpeg 加入 PATH")
        _FFMPEG = exe
    return _FFMPEG


def find_episodes():
    eps = []
    for name in os.listdir(MOVIE_DIR):
        m = re.match(r"第([\d.]+)话 .+\.mp4$", name)
        if m:
            label = m.group(1)
            eps.append((label, os.path.join(MOVIE_DIR, name)))
    return sorted(eps, key=lambda x: float(x[0]))


def op_ed_ranges(ep_no):
    """从该集元数据 JSON 中读 B 站官方 OP/ED 跳过区间(秒)."""
    metas = [f for f in os.listdir(MOVIE_DIR)
             if f.startswith(f"第{ep_no}话 ") and f.endswith("-元数据.json")]
    if not metas:
        return []
    with open(os.path.join(MOVIE_DIR, metas[0]), encoding="utf-8") as f:
        data = json.load(f)
    for ep in data.get("episodes", []):
        if ep.get("title") == str(ep_no):
            skip = ep.get("skip") or {}
            return [(skip[k]["start"], skip[k]["end"]) for k in ("op", "ed") if k in skip]
    return []


def prepare_episode(ep_no, mp4_path, ep_dir):
    """抽音轨/分离/转16k, 幂等, 返回 vocals, vocals16k 路径."""
    os.makedirs(ep_dir, exist_ok=True)
    audio = os.path.join(ep_dir, "audio.wav")
    vocals = os.path.join(ep_dir, "separated", "htdemucs", "audio", "vocals.wav")
    vocals16k = os.path.join(ep_dir, "vocals_16k.wav")
    if not os.path.exists(audio):
        subprocess.run([ffmpeg_exe(), "-y", "-hide_banner", "-loglevel", "error",
                        "-i", mp4_path, "-vn", "-ac", "2", "-ar", "44100", audio], check=True)
    if not os.path.exists(vocals):
        subprocess.run([".venv/bin/python", "-m", "demucs", "--two-stems=vocals",
                        "-n", "htdemucs", "--device", DEVICE,
                        "--out", os.path.join(ep_dir, "separated"), audio], check=True)
    if not os.path.exists(vocals16k):
        subprocess.run([ffmpeg_exe(), "-y", "-hide_banner", "-loglevel", "error",
                        "-i", vocals, "-ac", "1", "-ar", "16000", vocals16k], check=True)
    return vocals, vocals16k


def transcribe(vocals16k, seg_path, skip_ranges):
    if os.path.exists(seg_path):
        with open(seg_path, encoding="utf-8") as f:
            return json.load(f)
    audio, sr = sf.read(vocals16k)
    result = mlx_whisper.transcribe(
        audio.astype("float32"),
        path_or_hf_repo="mlx-community/whisper-large-v3-turbo",
        language="ja",
        condition_on_previous_text=False,
        no_speech_threshold=0.6,
    )
    segments = []
    for s in result["segments"]:
        text = s["text"].strip()
        if not text or s.get("no_speech_prob", 0) >= 0.8:
            continue
        mid = (s["start"] + s["end"]) / 2
        if any(a <= mid <= b for a, b in skip_ranges):
            continue
        segments.append({"start": round(s["start"], 2), "end": round(s["end"], 2), "text": text})
    ms.atomic_write_text(seg_path, json.dumps(segments, ensure_ascii=False, indent=1))
    return segments


def embed_episode(encoder, ep_dir):
    """计算并缓存一集的句级 embedding, 幂等. 返回 (kept, X); 无转写则 (None, None).

    kept.json 为时长 >=0.5s 的句段列表 [{start,end,text}], 与 embeddings.npy 行对齐;
    prepare_labeling.py / finalize.py / build_transcript_index.py 均依赖该文件.
    """
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
    ms.atomic_write_text(kept_path, json.dumps(kept, ensure_ascii=False, indent=1))
    return kept, X


def main():
    candidates_dir = os.path.join(BUILD_DIR, "candidates")
    review_dir = os.path.join(BUILD_DIR, "review")
    os.makedirs(candidates_dir, exist_ok=True)
    os.makedirs(review_dir, exist_ok=True)

    ref_global = np.load(REF_PATH)
    encoder = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb", run_opts={"device": "cpu"})

    all_rows = []
    for ep_no, mp4_path in find_episodes():
        ep_name = ep_dir_name(ep_no)
        ep_dir = os.path.join(BUILD_DIR, ep_name)
        print(f"===== 第{ep_no}话 =====", flush=True)
        vocals, vocals16k = prepare_episode(ep_no, mp4_path, ep_dir)
        segments = transcribe(vocals16k, os.path.join(ep_dir, "segments.json"), op_ed_ranges(ep_no))
        kept, X = embed_episode(encoder, ep_dir)
        if kept is None:
            print("  无可用句段", flush=True)
            continue

        # 参考向量 = 全局确认样本 + 本集纯"ちい"独白段 的加权平均
        local = [i for i, s in enumerate(kept) if CHI_PATTERN.match(s["text"])]
        ref = ref_global * N_REF_SAMPLES
        if local:
            ref = ref + X[local].sum(axis=0)
        ref /= (N_REF_SAMPLES + len(local))
        ref /= np.linalg.norm(ref)

        sims = X @ ref
        # 从 44.1k 人声轨导出高质量切片
        vocals_audio, sr_v = sf.read(vocals, dtype="float32")
        ep_rows = []
        n_chi, n_review = 0, 0
        for seg, sim in zip(kept, sims):
            dur = seg["end"] - seg["start"]
            if not (MIN_DUR <= dur <= MAX_DUR) or sim < REVIEW_LO:
                continue
            s, e = int(seg["start"] * sr_v), int(seg["end"] * sr_v)
            clip = vocals_audio[s:e].mean(axis=1)  # 立体声混单声道
            clip = resample_poly(clip, 1, 2)         # 44100 -> 22050
            name = clip_name(ep_name, seg["start"])
            accepted = sim >= THRESHOLD
            row = ms.make_row(file=name, ep=ep_name, start=seg["start"], end=seg["end"],
                              prob=sim, source=ms.SOURCE_CANDIDATE if accepted else ms.SOURCE_REVIEW,
                              text=seg["text"])
            if accepted:
                sf.write(os.path.join(candidates_dir, f"{name}.wav"), clip, 22050,
                         subtype="PCM_16")
                n_chi += 1
            else:
                sf.write(os.path.join(review_dir, f"{name}.wav"), clip, 22050,
                         subtype="PCM_16")
                n_review += 1
            ep_rows.append(row)
        ms.write_metadata_full(os.path.join(ep_dir, "candidates.csv"), ep_rows)
        all_rows.extend(ep_rows)
        print(f"  segments={len(kept)} local_chi={len(local)}"
              f" 候选={n_chi} 待复核={n_review}", flush=True)

    ms.write_metadata_full(os.path.join(BUILD_DIR, "metadata_full.csv"), all_rows)
    ms.write_metadata_full(os.path.join(BUILD_DIR, "review.csv"),
                           [r for r in all_rows if r["source"] == ms.SOURCE_REVIEW])
    n_acc = sum(1 for r in all_rows if r["source"] == ms.SOURCE_CANDIDATE)
    print(f"完成: 候选 {n_acc} 段 -> {candidates_dir}, "
          f"待复核 {len(all_rows) - n_acc} 段 -> {review_dir}")
    print("下一步: .venv/bin/python pipeline/prepare_labeling.py")
    print("注意: dataset/ 正式文件由 finalize.py 产出, 本脚本不触碰 dataset/")


if __name__ == "__main__":
    main()
