#!/usr/bin/env bash
# 순차 배치. render.sh 가 매 호출마다 락을 잡고 푼다 → 동시 실행 불가능.
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
RUN="$1"; shift
for k in "$@"; do
  "$ROOT/lib/render.sh" "$RUN" "$ROOT/prompts/subjects/$k.txt" "$k" || echo "계속 진행"
done
echo ALL-DONE
