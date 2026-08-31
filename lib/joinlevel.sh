#!/usr/bin/env bash
# 클립별 레벨을 맞춰 이어 붙인다.
#
# H3 는 클립마다 독립 생성되어 공유 라우드니스 기준이 없다. 두 클립이 우연히 맞을 이유가 없다.
#
# 두 가지를 반드시 지킨다.
#  1. **선형(고정 게인)** 으로만 맞춘다. 동적 정규화는 의도된 소멸·정적을 밀어 올려 연출을 죽인다.
#  2. **증폭 상한을 둔다.** 조용한 클립을 대사 수준까지 끌어올리면 잡음 바닥이 드러나 쇳소리가 된다.
#     (2026-08-30: -54.8 LUFS 인서트를 -18 로 맞추며 +33dB 증폭 → 없던 쇳소리를 만들어냈다)
#
# 영화 믹싱의 기본은 대사를 기준선에 고정하고 앰비언스를 그 아래 두는 것이다.
# 라벨마다 목표를 따로 준다:  <라벨>[:<목표LUFS>]
#
# 사용: lib/joinlevel.sh <run-dir> <out.mp4> <라벨>[:목표] ...
#       TARGET_I  기본 목표 (기본 -18)
#       MAX_GAIN  최대 증폭 dB (기본 12). 넘으면 상한까지만 올린다.
#       TAME      "1" 이면 조용한 클립(TAME_BELOW 아래)의 9kHz 이상을 깎는다.
#
# 쇳소리의 정체 (2026-08-30 측정): 조용한 앰비언스 클립은 9kHz 부터 고역이 과하다.
# 대사 클립의 자연스러운 기울기와 비교하면 12~14kHz 에서 +8~10dB 초과한다.
# 좁은 공진이 아니라 넓은 단이므로 노치가 아니라 하이셸프로 잡는다.
# highshelf=f=9000:g=-10 이 대사 클립 기울기에 가장 근접한다(초과 6.3dB → 0.6dB).
set -uo pipefail
source "$(dirname "$0")/env.sh"
RUN="$(cd "$1" && pwd)"; OUT="$2"; shift 2
DEF="${TARGET_I:--18}"; MAXG="${MAX_GAIN:-12}"
TMP=$(mktemp -d); L="$TMP/list.txt"; : > "$L"
printf "  %-6s %10s %10s %9s\n" 라벨 원본 목표 적용게인
for spec in "$@"; do
  k="${spec%%:*}"; t="${spec#*:}"; [ "$t" = "$spec" ] && t="$DEF"
  SRC="$RUN/out/$k.mp4"
  M=$("$FF" -hide_banner -nostats -i "$SRC" -af "loudnorm=I=$t:TP=-1.5:LRA=20:print_format=json" -f null - 2>&1 | awk '/^\{/,/^\}/')
  g(){ echo "$M" | grep "\"$1\"" | sed 's/.*: *"//;s/".*//'; }
  I=$(g input_i)
  # 요구 게인이 상한을 넘으면 목표를 낮춰 잡는다 — 잡음 바닥을 끌어올리지 않는다
  EFF=$(awk -v i="$I" -v t="$t" -v m="$MAXG" 'BEGIN{g=t-i; if(g>m) t=i+m; printf "%.2f",t}')
  GAIN=$(awk -v i="$I" -v e="$EFF" 'BEGIN{printf "%+.1f",e-i}')
  CAP=$(awk -v t="$t" -v e="$EFF" 'BEGIN{print (e<t-0.05)?" 상한적용":""}')
  printf "  %-6s %9s %10s %9s%s\n" "$k" "$I" "$EFF" "$GAIN" "$CAP"
  TAMEF=""
  if [ "${TAME:-0}" = "1" ]; then
    TB="${TAME_BELOW:--35}"
    if awk -v i="$I" -v b="$TB" 'BEGIN{exit !(i<b)}'; then
      TAMEF="highshelf=f=${TAME_F:-9000}:g=${TAME_G:--10},"
      printf "         └ 고역 완화 %sHz %sdB\n" "${TAME_F:-9000}" "${TAME_G:--10}"
    fi
  fi
  "$FF" -y -v error -i "$SRC" \
    -af "${TAMEF}loudnorm=I=$EFF:TP=-1.5:LRA=20:linear=true:measured_I=$I:measured_TP=$(g input_tp):measured_LRA=$(g input_lra):measured_thresh=$(g input_thresh),aresample=32000" \
    -c:v copy -c:a aac -b:a 192k "$TMP/$k.mp4"
  echo "file '$TMP/$k.mp4'" >> "$L"
done
"$FF" -y -v error -f concat -safe 0 -i "$L" -c copy "$OUT"
rm -rf "$TMP"; echo "→ $OUT"
