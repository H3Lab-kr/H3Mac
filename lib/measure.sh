#!/usr/bin/env bash
# 자동 계측기. 알고리즘으로 확실한 것만 잰다 — 화질 판정은 사람이 한다.
# 사용: lib/measure.sh <run-dir>
set -uo pipefail
source "$(dirname "$0")/env.sh"
RUN="$1"
printf "%-22s %8s %8s %8s %7s  %s\n" 파일 피크dBFS RMSdBFS 프레임간 길이초 경고
for f in "$RUN"/out/*.mp4; do
  b=$(basename "$f" .mp4)
  read pk rms <<< $("$FF" -hide_banner -nostats -i "$f" -af astats=measure_overall=Peak_level+RMS_level -f null - 2>&1 \
      | grep -oE "(Peak|RMS) level dB: -?[0-9.]+" | tail -2 | awk -F': ' '{printf "%s ",$2}')
  yd=$("$FF" -hide_banner -nostats -i "$f" -vf "tblend=all_mode=difference,signalstats,metadata=print:key=lavfi.signalstats.YAVG" -f null - 2>&1 \
      | grep -oE "YAVG=[0-9.]+" | awk -F= '{s+=$2;n++}END{if(n)printf "%.2f",s/n; else print "NA"}')
  d=$("$FP" -v error -show_entries format=duration -of csv=p=0 "$f")
  # 측정이 비면 0 이 아니라 NA 로 — 빈 값을 수치로 오해한 적이 있다
  [ -z "$pk" ] && pk=NA; [ -z "$rms" ] && rms=NA
  warn=""
  [ "$pk" != NA ] && warn=$(awk -v p="$pk" -v r="$rms" -v y="$yd" 'BEGIN{
    s=""; if(p>-0.5)s=s"클리핑 "; if(r<-70)s=s"무음 "; if(y!="NA"&&y+0>15)s=s"모션과다 "; if(y!="NA"&&y+0<0.15)s=s"정지 "; print s}')
  printf "%-22s %8s %8s %8s %7.2f  %s\n" "$b" "$pk" "$rms" "$yd" "$d" "$warn"
done
