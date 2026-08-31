#!/usr/bin/env python3
"""분신(고스팅) A/B 비교기.

겹친 상은 엣지를 두 벌 만들어 오프셋 10~14px 자기상관을 높인다.

⚠ 절대 임계값은 쓸 수 없다. 이 점수는 소재에 강하게 의존한다 —
제품 클로즈업(가죽·비누)은 정상인데도 0.5 가 나오고, 인물은 정상이 0.20 이다.
반드시 **같은 소재의 기준선과 비교**해야 한다. 소재가 다르면 무의미하다.
(2026-08-29: 절대 임계 0.24 로 시도했다가 36편 중 14건 오탐. 기각.)

사용: lib/ghost.py <기준선.mp4> <비교.mp4> [샘플수]
"""
import subprocess, sys, pathlib, numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
FF = str(ROOT / "bin" / "ffmpeg")

def frames(path, n=6, w=256):
    out = subprocess.run([FF, "-v", "error", "-i", str(path),
        "-vf", f"scale={w}:-2,format=gray,fps=source_fps", "-f", "rawvideo", "-"],
        capture_output=True)
    # 높이는 종횡비로 결정 — 실제 크기를 되읽는다
    probe = subprocess.run([str(ROOT/"bin"/"ffprobe"), "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=width,height", "-of", "csv=p=0", str(path)],
        capture_output=True, text=True).stdout.strip().split(",")
    W, H = int(probe[0]), int(probe[1])
    h = int(round(w * H / W)) // 2 * 2
    buf = np.frombuffer(out.stdout, dtype=np.uint8)
    k = len(buf) // (w * h)
    if k == 0: return []
    fr = buf[:k*w*h].reshape(k, h, w).astype(np.float32)
    idx = np.linspace(0, k-1, min(n, k)).astype(int)
    return [fr[i] for i in idx]

# 판별 대역: 256px 폭 기준 수평 오프셋 10~14px.
# 분신은 이 거리에서 엣지를 두 벌 만든다. (2026-08-29 tokenred 로 보정)
LO, HI = 10, 14

def ghost_score(f):
    gy, gx = np.gradient(f)
    e = np.hypot(gx, gy)
    e -= e.mean()
    F = np.fft.rfft2(e)
    ac = np.fft.fftshift(np.fft.irfft2(F * np.conj(F), s=e.shape))
    cy, cx = np.array(ac.shape) // 2
    peak = ac[cy, cx]
    if peak <= 0: return 0.0
    band = ac[cy-3:cy+4, :] / peak
    vals = [(band[:, cx+d].mean() + band[:, cx-d].mean()) / 2 for d in range(LO, HI+1)]
    return float(np.mean(vals))

def score(p, n=6):
    fs = frames(p, n)
    return float(np.mean([ghost_score(f) for f in fs])) if fs else float("nan")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        sys.exit("사용: lib/ghost.py <기준선.mp4> <비교.mp4> [샘플수]\n"
                 "     절대 임계는 소재 의존성 때문에 쓸 수 없다. 같은 소재끼리만 비교하라.")
    n = int(sys.argv[3]) if len(sys.argv) > 3 else 6
    a, b = score(sys.argv[1], n), score(sys.argv[2], n)
    r = b / a if a else float("nan")
    verdict = "분신 의심 — 눈으로 확인하라" if r > 1.20 else "특이 없음"
    print(f"  기준선 {pathlib.Path(sys.argv[1]).stem:14s} {a:.3f}")
    print(f"  비교   {pathlib.Path(sys.argv[2]).stem:14s} {b:.3f}   {r:.2f}배  {verdict}")
