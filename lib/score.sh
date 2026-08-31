#!/usr/bin/env bash
# 음악 베드를 영상에 얹는다. H3 원음을 살리고 음악은 그 밑으로 깐다.
#
# 설계 원칙
#  - 사이드체인 덕킹: 대사가 나올 때만 음악이 물러난다
#    임계를 낮게(0.03) 잡으면 전투처럼 큰 구간에서 음악이 통째로 짓눌린다.
#    0.14 / ratio 3.5 가 대사는 비켜가고 전투는 통과시키는 지점이다.
#  - 250Hz 하이패스: 목소리 대역을 비워준다
#  - 구간별 페이드: 막이 바뀌는 지점에서 베드를 교차시킨다
#  - 원본 오디오는 감쇠하지 않는다. 음악만 조절한다
set -uo pipefail
source "$(dirname "$0")/env.sh"
VID="$1"; OUT="$2"; shift 2   # 이후 인자: <베드>:<시작초>:<길이>:<페이드인>:<페이드아웃>[:<개별게인>]
MUSIC_GAIN="${MUSIC_GAIN:-0.55}"
TARGET_I="${TARGET_I:--16}"   # 시네마틱 -16 · 플랫폼 업로드 -14
TARGET_TP="${TARGET_TP:--1.5}" # 재인코딩 여유. 플랫폼용은 -1.5 이상 확보한다
LIMIT="${LIMIT:-0.94}"        # alimiter 상한(샘플 피크)
# alimiter 는 샘플 피크만 본다. AAC 인코딩이 인터샘플 피크를 만들어 트루 피크가 넘칠 수 있다.
# 4배 오버샘플링 구간에서 리미팅해 인터샘플 피크까지 잡는다.
OS_RATE="${OS_RATE:-176400}"
i=1; INS=(); FC=""; MIXIN=""
for spec in "$@"; do
  IFS=: read -r path st dur fi fo g <<< "$spec"
  g="${g:-1.0}"
  INS+=(-i "$path")
  FC+="[$i:a]atrim=0:$dur,asetpts=PTS-STARTPTS,volume=$g,afade=t=in:st=0:d=$fi,"
  FC+="afade=t=out:st=$(awk -v d=$dur -v f=$fo 'BEGIN{print d-f}'):d=$fo,"
  FC+="adelay=$(awk -v s=$st 'BEGIN{printf "%d",s*1000}')|$(awk -v s=$st 'BEGIN{printf "%d",s*1000}')[m$i];"
  MIXIN+="[m$i]"; i=$((i+1))
done
N=$((i-1))
FC+="${MIXIN}amix=inputs=$N:normalize=0:duration=longest[bed];"
FC+="[bed]highpass=f=250,volume=$MUSIC_GAIN[bedv];"
FC+="[bedv][0:a]sidechaincompress=threshold=${SC_THRESH:-0.14}:ratio=${SC_RATIO:-3.5}:attack=25:release=280:level_sc=1[duck];"
FC+="[0:a][duck]amix=inputs=2:normalize=0[premix];"
# PRECOMP=1 이면 loudnorm 전에 압축한다. 원본 LRA 가 20 을 넘으면 2패스 선형만으로는
# -14 LUFS 에 도달하지 못한다(단일 패스 동적 모드도 -16 대에서 멈춘다).
if [ "${PRECOMP:-0}" = "1" ]; then
  FC+="[premix]acompressor=threshold=${CMP_THRESH:-0.06}:ratio=${CMP_RATIO:-3}:attack=20:release=300:makeup=1[mix];"
else
  FC+="[premix]anull[mix];"
fi
# 1패스: 믹스 라우드니스를 잰다. 동적 loudnorm 은 조용한 구간을 밀어올려
# 시네마틱 다이내믹을 뭉갠다. 선형(linear=true)으로 고정 게인만 적용한다.
MEAS=$("$FF" -hide_banner -nostats -i "$VID" "${INS[@]}" \
  -filter_complex "${FC}[mix]loudnorm=I=$TARGET_I:TP=$TARGET_TP:LRA=20:print_format=json[a]" \
  -map "[a]" -f null - 2>&1 | awk '/^\{/,/^\}/')
gv(){ echo "$MEAS" | grep "\"$1\"" | sed 's/.*: *"//;s/".*//'; }
# LINEAR=1 (기본): 고정 게인만. 시네마틱 다이내믹을 보존한다.
# LINEAR=0        : 동적 압축. 넓은 LRA 를 목표 LRA 로 눌러 플랫폼 규격을 맞춘다.
#   원본 LRA 가 20 을 넘으면 선형으로는 -14 LUFS 와 -1.5 dBTP 를 동시에 못 맞춘다.
LN="loudnorm=I=$TARGET_I:TP=$TARGET_TP:LRA=${TARGET_LRA:-20}"
if [ "${LINEAR:-1}" = "1" ]; then
  LN="$LN:linear=true:measured_I=$(gv input_i):measured_TP=$(gv input_tp)"
  LN="$LN:measured_LRA=$(gv input_lra):measured_thresh=$(gv input_thresh)"
  MODE="선형(다이내믹 보존)"
else
  MODE="동적(플랫폼 규격)"
fi
echo "  측정 I=$(gv input_i) TP=$(gv input_tp) LRA=$(gv input_lra) → $MODE"
"$FF" -y -v error -i "$VID" "${INS[@]}" \
  -filter_complex "${FC}[mix]${LN},aresample=$OS_RATE,alimiter=limit=$LIMIT:level=disabled,aresample=32000[aout]" \
  -map 0:v -map "[aout]" -c:v copy -c:a aac -b:a 192k "$OUT"
echo "→ $OUT"
