#!/usr/bin/env bash
# 다음 실행 디렉터리를 만든다. 이름: <날짜>-<전역 일련번호>-<슬러그>
# 하루에 여러 건을 돌리므로 날짜만으로는 순서를 못 읽는다. 번호가 곧 실행 순서다.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SLUG="${1:?사용: lib/newrun.sh <슬러그>}"
N=$(ls -d "$ROOT"/runs/*-[0-9][0-9]-* 2>/dev/null | sed -E 's/.*-([0-9]{2})-.*/\1/' | sort -n | tail -1)
N=$(printf "%02d" $(( 10#${N:-0} + 1 )))
DIR="$ROOT/runs/$(date +%Y-%m-%d)-$N-$SLUG"
mkdir -p "$DIR/out"
echo "$DIR"
