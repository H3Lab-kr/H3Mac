#!/usr/bin/env bash
# 판정 세트 조립. 후처리를 넣지 않는다 — 사람이 볼 것은 H3 의 날것이다.
set -uo pipefail
source "$(dirname "$0")/env.sh"
RUN="$1"; shift
RUN="$(cd "$RUN" && pwd)"   # 절대·상대 경로 모두 허용
L="$RUN/.list.txt"; : > "$L"
for k in "$@"; do echo "file '$RUN/out/$k.mp4'" >> "$L"; done
"$FF" -y -v error -f concat -safe 0 -i "$L" -c copy "$RUN/JUDGE.mp4" \
  || "$FF" -y -v error -f concat -safe 0 -i "$L" -c:v libx264 -crf 16 -preset veryfast -pix_fmt yuv420p -c:a aac -b:a 192k "$RUN/JUDGE.mp4"
rm -f "$L"; echo "→ $RUN/JUDGE.mp4"
