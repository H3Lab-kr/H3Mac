#!/usr/bin/env bash
# 내부 렌더 해상도 A/B. 243프레임(10.1초) 두 편.
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
R="$(cat "$ROOT/runs/.render_res_run")"
P="$ROOT/prompts/homecoming/motion/h1c.txt"
S="$R/stills/h1c.png"
FIRST_FRAME="$S" "$ROOT/lib/render.sh" "$R" "$P" native 1024 576 4 50 243
sleep 120
FIRST_FRAME="$S" EXTRA="--reuse 1 --render-width 512 --render-height 288" \
  "$ROOT/lib/render.sh" "$R" "$P" half 1024 576 4 50 243
echo AB-RENDER-DONE
