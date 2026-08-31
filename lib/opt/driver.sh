#!/usr/bin/env bash
# 밤새 자율로 도는 최적화 하네스. 중단·재개 자유.
#
# 목표: 같은 품질에서 클립 한 편의 벽시계 시간을 최소화한다.
# 품질 가드: SSIM(화면) · 오디오 RMS · 9kHz 초과(쇳소리 대리지표)를 매 측정에서 기록한다.
#           속도만 보고 채택하면 망가진 조합을 고르게 된다.
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
TSV="$ROOT/runs/opt/results.tsv"; L="$ROOT/lib/opt"
t(){ "$L/trial.sh" "$@"; }
base(){ awk -F'\t' 'NR>1&&$2~/^base/&&$4!="NA"{s+=$4;n++}END{if(n)printf "%.2f",s/n}' "$TSV"; }

echo "════ 1단계 · 개별 손잡이 (31회) ════"
"$L/sweep.sh" "$L/plan.tsv"

echo "════ 2단계 · 상위 조합 ════"
B=$(base)
WIN=$(awk -F'\t' -v b="$B" 'NR>1&&$2!~/^base/&&$4!="NA"&&$7=="ok"&&($4-b)/b<-0.03{printf "%s\t%s\n",$4,$3}' "$TSV" \
      | sort -n | head -4 | cut -f2)
if [ -n "$WIN" ]; then
  ALL=$(echo "$WIN" | tr '\n' ' ')
  echo "  이긴 손잡이: $ALL"
  t "combo-all" "$ALL"
  i=1; for w in $WIN; do
    REST=$(echo "$WIN" | grep -vxF "$w" | tr '\n' ' ')
    [ -n "$REST" ] && t "combo-drop$i" "$REST"; i=$((i+1))
  done
else echo "  3% 넘게 이긴 손잡이 없음 — 조합 생략"; fi
t "base-7" ""

echo "════ 3단계 · 샘플러 공간 (품질 위험 큼) ════"
EXTRA="--core-reuse 2" t "s-corereuse2" ""
EXTRA="--core-reuse 4" t "s-corereuse4" ""
EXTRA="--core-reuse 6" t "s-corereuse6" ""
EXTRA="--reuse 2"      t "s-reuse2" ""
EXTRA="--reuse 3"      t "s-reuse3" ""
t "base-8" ""

echo "════ 4단계 · 렌더 해상도 낮추기 ════"
# 16:9 에서 양변이 32 배수인 하위 캔버스는 512x288 하나뿐이다(1024x576 의 1/4 픽셀).
EXTRA="--reuse 1 --render-width 512 --render-height 288"  t "r-512x288" ""
t "base-9" ""

echo "════ 5단계 · 최종 후보 반복 검증 (124프레임 실사이즈) ════"
# 서식이 붙은 manifest 문자열을 그대로 넘기면 안 된다(2026-08-31 버그).
# 최종 검증은 plan.tsv 의 원래 환경변수를 다시 찾아 쓴다.
TOP=$(awk -F'\t' 'NR>1&&$4!="NA"&&$7=="ok"&&$2!~/^base/&&$2!~/^s-/{printf "%s\t%s\n",$4,$2}' "$TSV" \
      | sort -n | head -2 | cut -f2)
for lbl in $TOP; do
  envs=$(awk -F'\t' -v l="$lbl" '$1==l{print $2}' "$L/plan.tsv")
  for r in 1 2; do F=124 t "final-${lbl}-r$r" "$envs"; done
done
for r in 1 2; do F=124 t "final-base-r$r" ""; done
echo DRIVER-DONE
