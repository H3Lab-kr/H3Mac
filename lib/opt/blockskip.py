#!/usr/bin/env python3
"""스텝 간 블록 변화량 분석 — 스킵 전략의 근거를 만든다.

입력: H3_BLOCK_PROBE=1 로 렌더한 stderr 로그
질문: 뒤 스텝에서 변화가 작은 블록이 정말 많은가? 몇 %를 건너뛸 수 있나?
사용: lib/opt/blockskip.py <로그>
"""
import re, sys, statistics as S
from collections import defaultdict

rows = []
for m in re.finditer(r"blockprobe: step=(\d+) block=(\d+) video=([\d.]+) "
                     r"sink=([\d.]+) vnorm=([\d.eE+-]+)",
                     open(sys.argv[1]).read()):
    rows.append((int(m.group(1)), int(m.group(2)), float(m.group(3)),
                 float(m.group(4)), float(m.group(5))))
if not rows:
    sys.exit("프로브 데이터 없음")

steps = sorted({r[0] for r in rows})
blocks = sorted({r[1] for r in rows})
print(f"  표본 {len(rows)}개 · 스텝 {len(steps)} × 블록 {len(blocks)}\n")

by_step = defaultdict(list)
sink_step = defaultdict(list)
for r in rows:
    by_step[r[0]].append(r[2])
    sink_step[r[0]].append(r[3])
print("  (video = 비디오 타깃 영역 · sink = 텍스트·오디오 조건 영역)")
print(f"  {'스텝':<6}{'비디오중앙':>12}{'비디오평균':>12}{'최대':>10}{'싱크중앙':>12}")
for st in steps:
    v, sk = by_step[st], sink_step[st]
    print(f"  {st:<6}{S.median(v):12.4f}{S.mean(v):12.4f}{max(v):10.4f}"
          f"{S.median(sk):12.4f}")

print(f"\n  ── 임계값별 스킵 가능 비율 (전체 {len(rows)}개 기준) ──")
print(f"  {'임계':<10}{'스킵가능':>10}{'비율':>8}   스텝별 분포")
for th in (0.001, 0.005, 0.01, 0.02, 0.05):
    hits = [r for r in rows if r[2] < th]
    dist = " ".join(f"s{st}:{sum(1 for h in hits if h[0]==st)}" for st in steps)
    print(f"  <{th:<9.3f}{len(hits):>10}{len(hits)/len(rows):>7.0%}   {dist}")

print(f"\n  ── 블록대별 평균 변화 (스텝 무관) ──")
by_block = defaultdict(list)
for r in rows:
    by_block[r[1]].append(r[2])
lo_hi = 10
for lo in range(0, max(blocks)+1, lo_hi):
    grp = [c for b in range(lo, lo+lo_hi) for c in by_block.get(b, [])]
    if grp:
        print(f"  블록 {lo:2d}~{lo+lo_hi-1:2d}   {S.mean(grp):.4f}")
