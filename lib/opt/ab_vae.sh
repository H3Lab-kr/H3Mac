#!/usr/bin/env bash
# 순정 vs bf16 VAE 포크. 냉각을 넣고 교차로 두 번씩 잰다(발열 드리프트 상쇄).
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
R="$ROOT/runs/2026-08-30-20-homecoming"
P="$(tr '\n' ' ' < "$ROOT/prompts/homecoming/motion/h1c.txt")"
run(){ # <엔진디렉터리> <출력>
  sleep 120
  cd "$ROOT/engines/$1"
  H3_PROFILE=1 H3_FFMPEG="$ROOT/bin/ffmpeg" H3_FFPROBE="$ROOT/bin/ffprobe" \
  ./h3 -d "$ROOT/models/MiniMax-H3-turbo4" -p "$P" \
    --first-frame "$R/stills/h1c.png" --width 1024 --height 576 --frames 73 \
    --steps 4 --layers 50 --reuse 1 --seed 42 -o "$2" 2>&1 \
    | tr '\r' '\n' | grep -E "video VAE decoder|GPU Euler denoise" \
    | sed "s|^|  $1  |"
}
for round in 3 4 5 6 7 8; do
  echo "── 라운드 $round ──"
  run h3.c        "$ROOT/runs/opt/mp4/vae_stock_r$round.mp4"
  run h3.c-bf16vae "$ROOT/runs/opt/mp4/vae_bf16_r$round.mp4"
done
echo AB-VAE-DONE
