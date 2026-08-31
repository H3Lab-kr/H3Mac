# 모든 스크립트가 이 파일 하나만 참조한다.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export ROOT
# 기본 엔진은 bf16 VAE 포크다 (2026-08-31 승격 · docs/VERDICTS.md V-009).
# 비디오 VAE 디코더가 3~5배 빠르고 피크 메모리가 절반이며, 화면·시간축 모두 무손실이다.
# 셰이더가 CWD 상대라 반드시 해당 디렉터리에서 실행해야 한다 — render.sh 가 처리한다.
export H3_DIR="${H3_DIR:-$ROOT/engines/h3.c-bf16vae}"
export H3_DIR_STOCK="$ROOT/engines/h3.c"   # 순정 업스트림 — 대조 실험용
export H3_MODEL="${H3_MODEL:-$ROOT/models/MiniMax-H3-turbo4}"   # 기준 모델 (2026-08-29 판정)
export H3_MODEL_BASE="$ROOT/models/MiniMax-H3"                  # 순정 — 대조용
export H3_FFMPEG="$ROOT/bin/ffmpeg"
export H3_FFPROBE="$ROOT/bin/ffprobe"
FF="$H3_FFMPEG"; FP="$H3_FFPROBE"; export FF FP
# GPU 는 하나뿐이다. 동시 실행은 측정을 오염시킨다.
h3_lock() {
  LOCKD="$ROOT/.h3.lock.d"
  mkdir "$LOCKD" 2>/dev/null || { echo "이미 실행 중"; return 1; }
  trap 'rmdir "$LOCKD" 2>/dev/null' EXIT INT TERM
  pgrep -x h3 >/dev/null && { echo "다른 h3 가 GPU 사용 중 — 중단"; return 1; }
  return 0
}
