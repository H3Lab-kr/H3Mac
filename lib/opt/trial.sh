#!/usr/bin/env bash
# 한 번의 측정. 환경변수 조합 하나를 재고 results.tsv 에 한 줄 남긴다.
#
# 규칙
#  - 측정 전 냉각한다. 발열은 성능 차이로 위장한다.
#  - 결과는 즉시 기록한다. 중간에 죽어도 지금까지가 남는다.
#  - 실패해도 기록한다. "이 조합은 안 된다"도 결과다.
#
# 사용: lib/opt/trial.sh <라벨> <환경변수 문자열>
#   예: lib/opt/trial.sh nax-mlp "H3_NAX=mlp"
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
LABEL="$1"; ENVS="${2:-}"
OUT="$ROOT/runs/opt"; TSV="$OUT/results.tsv"
COOL="${COOL:-120}"
# 탐색용 고정 조건 — 짧지만 대표성이 있어야 한다.
# 1024x576 · 73프레임이면 토큰 10,944 로 int8 게이트(>=128)를 넘고 어텐션도 유의미하다.
W=${W:-1024} H=${H:-576} F=${F:-73} STEPS=${STEPS:-4} LAYERS=${LAYERS:-50}
[ -f "$ROOT/lib/opt/paths.sh" ] && . "$ROOT/lib/opt/paths.sh"
STILL="${STILL:-$OPT_STILL}"
PROMPT="${PROMPT:-$OPT_PROMPT}"

[ -f "$TSV" ] || printf "시각\t라벨\t환경변수\tDiT초\tVAE초\t전체초\t상태\tSSIM\t오디오RMS\t고역초과\t산출물\n" > "$TSV"
mkdir -p "$OUT/mp4"
sleep "$COOL"

cd "$ROOT/engines/h3.c"
MP4="$OUT/mp4/$LABEL.mp4"
LOG=$(mktemp)
T0=$(date +%s)
env H3_FFMPEG="$ROOT/bin/ffmpeg" H3_FFPROBE="$ROOT/bin/ffprobe" H3_PROFILE=1 $ENVS \
  ./h3 -d "$ROOT/models/MiniMax-H3-turbo4" -p "$(tr '\n' ' ' < "$PROMPT")" \
  --first-frame "$STILL" --width "$W" --height "$H" --frames "$F" \
  --steps "$STEPS" --layers "$LAYERS" --seed 42 ${EXTRA:---reuse 1} \
  -o "$MP4" >"$LOG" 2>&1
RC=$?
T1=$(date +%s)
g(){ grep "$1" "$LOG" | grep -oE 'wall= *[0-9.]+' | head -1 | grep -oE '[0-9.]+'; }
DIT=$(g "GPU Euler denoise"); VAE=$(g "video VAE decoder")
ST="ok"; [ $RC -ne 0 ] && ST="실패"; [ -f "$MP4" ] || ST="산출물없음"
# ── 품질 계측 ── 속도만 보면 망가진 조합을 채택하게 된다.
REF="$OUT/mp4/ref.mp4"
[ -f "$REF" ] || { [ "$ST" = ok ] && cp "$MP4" "$REF"; }
SSIM=NA; ARMS=NA; HFX=NA
if [ "$ST" = ok ] && [ -f "$REF" ]; then
  SSIM=$("$ROOT/bin/ffmpeg" -hide_banner -nostats -i "$MP4" -i "$REF" -lavfi ssim -f null - 2>&1 \
        | grep -oE "All:[0-9.]+" | head -1 | cut -d: -f2)
  ARMS=$("$ROOT/bin/ffmpeg" -hide_banner -nostats -i "$MP4" -af astats=measure_overall=RMS_level -f null - 2>&1 \
        | grep -oE "RMS level dB: -?[0-9.]+" | tail -1 | awk -F': ' '{printf "%.1f",$2}')
  # 9kHz 이상 과잉 = 쇳소리 대리지표. 전체 대비 상대값.
  HI=$("$ROOT/bin/ffmpeg" -hide_banner -nostats -i "$MP4" -af "highpass=f=9000,astats=measure_overall=RMS_level" -f null - 2>&1 \
        | grep -oE "RMS level dB: -?[0-9.]+" | tail -1 | awk -F': ' '{printf "%.1f",$2}')
  HFX=$(awk -v a="$ARMS" -v h="$HI" 'BEGIN{if(a!=""&&h!="")printf "%.1f",h-a; else print "NA"}')
fi
printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
  "$(date +%H:%M:%S)" "$LABEL" "${ENVS:-기본}${EXTRA:+ | $EXTRA}${F:+ | ${F}f}" "${DIT:-NA}" "${VAE:-NA}" "$((T1-T0))" "$ST" \
  "${SSIM:-NA}" "${ARMS:-NA}" "${HFX:-NA}" "$MP4" >> "$TSV"
printf "  %-20s DiT %-7s 전체 %4d초  SSIM %-8s 고역 %-6s %s\n" "$LABEL" "${DIT:-NA}" "$((T1-T0))" "${SSIM:-NA}" "${HFX:-NA}" "$ST"
cp "$LOG" "$OUT/mp4/$LABEL.log"; rm -f "$LOG"
