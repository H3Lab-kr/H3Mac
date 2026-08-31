#!/usr/bin/env bash
# 순위표. 언제든 부를 수 있다.
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
TSV="$ROOT/runs/opt/results.tsv"
[ -f "$TSV" ] || { echo "  아직 결과 없음"; exit 0; }
echo "── 기준선 (발열 드리프트 검출) ──"
awk -F'\t' 'NR>1&&$2~/^base/&&$4!="NA"{printf "  %-10s DiT %7.1f  전체 %4d초\n",$2,$4,$6; s+=$4;n++; if(mn==""||$4<mn)mn=$4; if($4>mx)mx=$4}
END{if(n){printf "  평균 %.1f초 · 편차 %.1f~%.1f (%.1f%%)\n",s/n,mn,mx,(mx-mn)/mn*100
  printf "  → 이보다 작은 차이는 잡음으로 본다\n"}}' "$TSV"
B=$(awk -F'\t' 'NR>1&&$2~/^base/&&$4!="NA"{s+=$4;n++}END{if(n)printf "%.3f",s/n}' "$TSV")
SP=$(awk -F'\t' 'NR>1&&$2~/^base/&&$4!="NA"{if(mn==""||$4<mn)mn=$4; if($4>mx)mx=$4}END{if(mn)printf "%.1f",(mx-mn)/mn*100}' "$TSV")
echo
echo "── 순위 (DiT 시간 기준 · 잡음 폭 ${SP:-?}% 초과분만 의미 있음) ──"
printf "  %-22s %8s %8s %7s %8s %7s\n" 라벨 DiT 전체 대비 SSIM 고역
awk -F'\t' -v b="$B" 'NR>1&&$4!="NA"&&$2!~/^base/{
  d=($4-b)/b*100
  printf "%9.3f\t  %-22s %8.1f %7d초 %+6.1f%% %8s %6s %s\n",$4,$2,$4,$6,d,$8,$10,($7!="ok"?"["$7"]":"")
}' "$TSV" | sort -n | cut -f2- | head -22
echo
echo "── 총 측정 $(awk 'END{print NR-1}' "$TSV")회 ──"
