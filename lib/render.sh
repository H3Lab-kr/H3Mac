#!/usr/bin/env bash
# 단일 진입점. 지난번 실패는 거의 같은 스크립트 8개 복제본이었다 — 여기 하나만 둔다.
# 사용: lib/render.sh <run-dir> <prompt.txt> <라벨> [W H STEPS LAYERS FRAMES]
set -uo pipefail
source "$(dirname "$0")/env.sh"
RUN="$1"; PROMPT="$2"; LABEL="$3"
mkdir -p "$RUN"
RUN="$(cd "$RUN" && pwd)"          # cd "$H3_DIR" 이후에도 유효하도록 절대경로로
PROMPT="$(cd "$(dirname "$PROMPT")" && pwd)/$(basename "$PROMPT")"
W=${4:-576} H=${5:-1024} STEPS=${6:-4} LAYERS=${7:-50} FRAMES=${8:-73} SEED=${SEED:-42}
mkdir -p "$RUN/out"
MAN="$RUN/manifest.tsv"
[ -f "$MAN" ] || printf "라벨\t가로\t세로\t프레임\t스텝\t레이어\t시드\t초\t추가옵션\t첫프레임\t파일\n" > "$MAN"
OUT="$RUN/out/$LABEL.mp4"
[ -f "$OUT" ] && { echo "SKIP $LABEL"; exit 0; }
h3_lock || exit 1
cd "$H3_DIR"
t0=$(date +%s)
# HALF=1 이면 내부 계산을 절반 해상도에서 한다(출력 크기는 그대로).
# 토큰이 1/4 이라 DiT 가 급격히 빨라진다. 대가는 확대 시 선명도 — V-011.
# 양변이 32 의 배수여야 하고 종횡비가 같아야 한다.
if [ "${HALF:-0}" = "1" ] && [ -z "${EXTRA:-}" ]; then
  RW=$(( W / 2 )); RH=$(( H / 2 ))
  if [ $((RW % 32)) -eq 0 ] && [ $((RH % 32)) -eq 0 ]; then
    EXTRA="--reuse 1 --render-width $RW --render-height $RH"
  else
    echo "  HALF 무시: ${RW}x${RH} 가 32 배수가 아니다"
  fi
fi
FF_ARGS=()
[ -n "${FIRST_FRAME:-}" ] && FF_ARGS+=(--first-frame "$FIRST_FRAME")
# FL2VA: 끝 프레임까지 지정하면 클립이 그 이미지에서 끝난다.
# 다음 클립의 첫 프레임을 여기 주면 이음매가 구조적으로 사라진다.
[ -n "${LAST_FRAME:-}" ] && FF_ARGS+=(--last-frame "$LAST_FRAME")
./h3 -d "$H3_MODEL" -p "$(tr '\n' ' ' < "$PROMPT")" ${FF_ARGS[@]+"${FF_ARGS[@]}"} \
     --width "$W" --height "$H" --frames "$FRAMES" \
     --steps "$STEPS" --layers "$LAYERS" --seed "$SEED" ${EXTRA:---reuse 1} \
     -o "$OUT" >"$RUN/out/$LABEL.log" 2>&1
dt=$(( $(date +%s) - t0 ))
if [ -f "$OUT" ]; then
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$LABEL" "$W" "$H" "$FRAMES" "$STEPS" "$LAYERS" "$SEED" "$dt" "${EXTRA:-–reuse 1}" "${FIRST_FRAME:+$(basename "$FIRST_FRAME")}${LAST_FRAME:+→$(basename "$LAST_FRAME")}" "out/$LABEL.mp4" >> "$MAN"
  echo "DONE $LABEL ${dt}초"
else echo "FAIL $LABEL — 로그: $RUN/out/$LABEL.log"; exit 1; fi
