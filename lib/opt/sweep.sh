#!/usr/bin/env bash
# 계획서(plan.tsv)를 순서대로 돌린다. 중단해도 이어서 할 수 있다.
#
# 기준선(base-N)을 중간중간 반복 측정한다. 그 값들이 서로 크게 다르면
# 그 사이 측정은 발열 드리프트에 오염된 것이므로 믿지 않는다.
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
PLAN="${1:-$ROOT/lib/opt/plan.tsv}"
TSV="$ROOT/runs/opt/results.tsv"
tail -n +2 "$PLAN" | while IFS=$'\t' read -r LABEL ENVS; do
  [ -z "$LABEL" ] && continue
  # 이미 측정한 라벨은 건너뛴다 → 중단·재개가 자유롭다
  if [ -f "$TSV" ] && awk -F'\t' -v l="$LABEL" 'NR>1&&$2==l{f=1}END{exit !f}' "$TSV"; then
    echo "  SKIP $LABEL"; continue
  fi
  "$ROOT/lib/opt/trial.sh" "$LABEL" "$ENVS"
done
echo SWEEP-DONE
