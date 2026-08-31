#!/usr/bin/env python3
"""시간축 떨림 탐지기.

프레임 단위 SSIM 은 떨림을 못 잡는다 — 각 프레임이 기준과 비슷해도
프레임 사이가 들썩이면 눈에는 떨림으로 보인다. (2026-08-31 core-reuse 6 에서 놓쳤다)

두 지표를 낸다.
  밝기요동  프레임 평균 휘도의 프레임간 변화. 전역 펄싱을 잡는다.
  가속도    프레임간 차분의 2차 변화. 실제 움직임은 매끄럽고 떨림은 각지다.
⚠ 절대 임계값은 쓸 수 없다. 내용에 따라 가속도가 5(정지 인물)에서 59(아니메 전투)까지 간다.
   반드시 **같은 프롬프트·같은 스틸로 만든 기준선과 비교**해야 한다.
   (2026-08-31: 고스팅 탐지기 V-004 와 똑같은 함정. 정상 표본 9편으로 확인했다.)

사용: lib/opt/flicker.py <기준선.mp4> <비교.mp4>
"""
import subprocess, sys, re, pathlib
import numpy as np
ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
FF = str(ROOT/"bin"/"ffmpeg"); FP = str(ROOT/"bin"/"ffprobe")

def frames(path, w=160):
    p = subprocess.run([FP,"-v","error","-select_streams","v:0",
        "-show_entries","stream=width,height","-of","csv=p=0",path],
        capture_output=True,text=True).stdout.strip().split(",")
    W,H = int(p[0]), int(p[1]); h = int(round(w*H/W))//2*2
    r = subprocess.run([FF,"-v","error","-i",path,"-vf",f"scale={w}:{h},format=gray",
        "-f","rawvideo","-"],capture_output=True)
    b = np.frombuffer(r.stdout,dtype=np.uint8)
    n = len(b)//(w*h)
    return b[:n*w*h].reshape(n,h,w).astype(np.float32)

def metrics(path):
    f = frames(path)
    if len(f) < 4: return None
    mean = f.reshape(len(f),-1).mean(1)          # 프레임 평균 휘도
    dmean = np.abs(np.diff(mean))
    bright = float(dmean.std() / (mean.mean()+1e-6) * 100)
    d = np.abs(np.diff(f,axis=0)).reshape(len(f)-1,-1).mean(1)   # 프레임간 차분
    accel = float(np.abs(np.diff(d)).mean() / (d.mean()+1e-6) * 100)
    return bright, accel, float(d.mean())

if __name__ == "__main__":
    if len(sys.argv) < 3:
        sys.exit("사용: lib/opt/flicker.py <기준선.mp4> <비교.mp4>\n"
                 "     절대 임계는 내용 의존성 때문에 쓸 수 없다. 같은 소재끼리만 비교하라.")
    a, b = metrics(sys.argv[1]), metrics(sys.argv[2])
    if not a or not b: sys.exit("측정 불가")
    print(f"  {'':<14}{'밝기요동':>9}{'가속도':>9}{'프레임간':>9}")
    print(f"  {'기준선':<14}{a[0]:9.2f}{a[1]:9.1f}{a[2]:9.2f}")
    print(f"  {'비교':<14}{b[0]:9.2f}{b[1]:9.1f}{b[2]:9.2f}")
    rb, ra = b[0]/max(a[0],1e-6), b[1]/max(a[1],1e-6)
    print(f"  {'배율':<14}{rb:9.2f}{ra:9.2f}")
    if ra > 1.3 or rb > 1.3:
        print("\n  떨림 의심 — 눈으로 확인하라")
    else:
        print("\n  특이 없음")
