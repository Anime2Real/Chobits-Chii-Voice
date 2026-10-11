#!/usr/bin/env bash
# 人声分离: demucs htdemucs, 只保留 vocals 轨
# 用法: pipeline/legacy/02_separate.sh pilot/ep01/audio.wav pilot/ep01
set -euo pipefail

IN="$1"
OUTDIR="$2"

.venv/bin/python -m demucs \
  --two-stems=vocals \
  -n htdemucs \
  --device "${DEVICE:-mps}" \
  --out "$OUTDIR/separated" \
  "$IN"

VOCALS="$OUTDIR/separated/htdemucs/$(basename "${IN%.*}")/vocals.wav"
# 转为 16kHz 单声道, 方便后续 ASR / embedding
FF=$(.venv/bin/python -c "import imageio_ffmpeg; print(imageio_ffmpeg.get_ffmpeg_exe())")
"$FF" -y -hide_banner -loglevel error -i "$VOCALS" -ac 1 -ar 16000 "$OUTDIR/vocals_16k.wav"
echo "OK: $OUTDIR/vocals_16k.wav"
