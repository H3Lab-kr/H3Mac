#!/usr/bin/env python3
"""레퍼런스 스틸 생성. OpenAI 이미지 → 세로 영상 캔버스에 맞게 크롭.

사용: lib/imagegen.py <out.png> <프롬프트파일> [--w 512] [--h 896] [--model gpt-image-1-mini]
"""
import os, sys, json, base64, urllib.request, subprocess, time, argparse, pathlib

ROOT = pathlib.Path(__file__).resolve().parent.parent
FFMPEG = str(ROOT / "bin" / "ffmpeg")

# OpenAI 세로 사이즈는 1024x1536 (2:3) 뿐이다. 9:16 영상보다 넓으므로 가로를 잘라낸다.
GEN_W, GEN_H = 1024, 1536          # 세로 기본. --landscape 로 1536x1024

def load_key():
    if key := os.environ.get("OPENAI_API_KEY"):
        return key.strip()
    env_file = ROOT / ".env"
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            if line.startswith("OPENAI_API_KEY"):
                return line.split("=", 1)[1].strip().strip('"').strip("'")
    raise SystemExit("OPENAI_API_KEY 환경변수 또는 .env 파일이 필요합니다.")

def generate(prompt, model, raw_path, quality="standard"):
    body = {"model": model, "prompt": prompt, "size": f"{GEN_W}x{GEN_H}", "n": 1}
    if quality:
        body["quality"] = quality
    req = urllib.request.Request(
        "https://api.openai.com/v1/images/generations",
        data=json.dumps(body).encode(),
        headers={"Authorization": f"Bearer {load_key()}", "Content-Type": "application/json"})
    t0 = time.time()
    try:
        d = json.load(urllib.request.urlopen(req, timeout=300))
    except urllib.error.HTTPError as e:
        sys.exit(f"HTTP {e.code}: {e.read().decode()[:400]}")
    raw_path.write_bytes(base64.b64decode(d["data"][0]["b64_json"]))
    u = d.get("usage", {})
    return time.time() - t0, u.get("output_tokens", 0)

def fit(raw_path, out_path, w, h):
    """생성 원본을 목표 종횡비로 센터 크롭한 뒤 리사이즈."""
    gw, gh = GEN_W, GEN_H
    cw = round(gh * w / h)
    if cw <= gw:                         # 목표가 더 좁다 → 가로를 자른다
        vf = f"crop={cw}:{gh}:{(gw-cw)//2}:0,scale={w}:{h}:flags=lanczos"
    else:                                # 목표가 더 넓다 → 세로를 자른다
        ch = round(gw * h / w)
        vf = f"crop={gw}:{ch}:0:{(gh-ch)//2},scale={w}:{h}:flags=lanczos"
    subprocess.run([FFMPEG, "-y", "-v", "error", "-i", str(raw_path),
                    "-vf", vf, str(out_path)], check=True)
    return vf

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("out"); ap.add_argument("prompt_file")
    ap.add_argument("--w", type=int, default=1024); ap.add_argument("--h", type=int, default=576)
    ap.add_argument("--model", default="gpt-image-2-mini", help="이미지 모델 (기본: gpt-image-2-mini)")
    ap.add_argument("--quality", default="standard", help="화질 (standard, low)")
    ap.add_argument("--landscape", action="store_true", default=True, help="1536x1024 가로 생성 (기본값)")
    ap.add_argument("--portrait", action="store_true", help="1024x1536 세로 생성")
    a = ap.parse_args()
    if a.portrait:
        globals()["GEN_W"], globals()["GEN_H"] = 1024, 1536
        if a.w == 1024 and a.h == 576:
            a.w, a.h = 576, 1024
    else:
        globals()["GEN_W"], globals()["GEN_H"] = 1536, 1024

    out = pathlib.Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    raw = out.with_name(out.stem + "_raw.png")
    prompt = pathlib.Path(a.prompt_file).read_text().strip()

    dt, tok = generate(prompt, a.model, raw)
    vf = fit(raw, out, a.w, a.h)
    print(f"{out.name}  {dt:.1f}초  출력토큰 {tok}  {a.w}x{a.h}")
    # 이미지도 설정을 옆에 남긴다
    (out.parent / "images.tsv").open("a").write(
        f"{out.stem}\t{a.model}\t{GEN_W}x{GEN_H}\t{a.w}x{a.h}\t{tok}\t{dt:.1f}\n")
