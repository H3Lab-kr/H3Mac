#!/usr/bin/env bash
# 야간 실행 진입점. 저장소 안에 있으므로 재부팅에도 살아남는다.
cd "$(cd "$(dirname "$0")/../.." && pwd)"
exec ./lib/opt/driver.sh
