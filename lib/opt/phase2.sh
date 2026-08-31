#!/usr/bin/env bash
# 1차 하네스에서 놓친 것들을 다시 잰다.
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"; L="$ROOT/lib/opt"
t(){ "$L/trial.sh" "$@"; }
t "base-10" ""
EXTRA="--reuse 1 --render-width 512 --render-height 288" t "r-512x288" ""
t "base-11" ""
# corereuse6 을 124프레임 실사이즈로 재확인 (73프레임 결과가 유지되는지)
F=124 EXTRA="--core-reuse 6" t "final-corereuse6-r1" ""
F=124 t "final-base124-r1" ""
F=124 EXTRA="--core-reuse 6" t "final-corereuse6-r2" ""
F=124 t "final-base124-r2" ""
echo PHASE2-DONE
