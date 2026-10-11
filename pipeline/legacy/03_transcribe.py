#!/usr/bin/env python3
"""ASR 转写: mlx-whisper large-v3-turbo, 日语, 输出带时间戳的 segments JSON.

用法: .venv/bin/python pipeline/legacy/03_transcribe.py pilot/ep01/vocals_16k.wav pilot/ep01/segments.json
"""
import json
import sys

import mlx_whisper
import soundfile as sf


def main():
    wav, out = sys.argv[1], sys.argv[2]
    audio, sr = sf.read(wav)
    assert sr == 16000, f"need 16kHz mono wav, got sr={sr}"
    result = mlx_whisper.transcribe(
        audio.astype("float32"),
        path_or_hf_repo="mlx-community/whisper-large-v3-turbo",
        language="ja",
        condition_on_previous_text=False,
        no_speech_threshold=0.6,
    )
    segments = [
        {"start": round(s["start"], 2), "end": round(s["end"], 2), "text": s["text"].strip()}
        for s in result["segments"]
        if s.get("no_speech_prob", 0) < 0.8 and s["text"].strip()
    ]
    with open(out, "w", encoding="utf-8") as f:
        json.dump(segments, f, ensure_ascii=False, indent=1)
    print(f"{len(segments)} segments -> {out}")


if __name__ == "__main__":
    main()
