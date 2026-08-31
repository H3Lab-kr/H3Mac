#!/usr/bin/env python3
"""히어로 스틸 한 장에서 다른 컷의 스틸을 파생시킨다.

항상 히어로 원본을 편집한다 — 이전 결과를 다시 편집하면 얼굴이 서서히 흘러간다.
사용: lib/imageedit.py <히어로_raw.png> <out.png> <프롬프트파일> [--w 512] [--h 896]
"""
import os, sys, json, base64, urllib.request, uuid, time, argparse, pathlib, subprocess

ROOT = pathlib.Path(__file__).resolve().parent.parent
FFMPEG = str(ROOT / "bin" / "ffmpeg")
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

def edit(hero, prompt, model, raw_out):
    b = uuid.uuid4().hex
    def part(n, v):
        return f'--{b}\r\nContent-Disposition: form-data; name="{n}"\r\n\r\n{v}\r\n'.encode()
    body = part("model", model) + part("prompt", prompt) + part("size", f"{GEN_W}x{GEN_H}")
    body += (f'--{b}\r\nContent-Disposition: form-data; name="image"; filename="i.png"\r\n'
             f'Content-Type: image/png\r\n\r\n').encode() + hero.read_bytes() + b"\r\n"
    body += f"--{b}--\r\n".encode()
    req = urllib.request.Request("https://api.openai.com/v1/images/edits", data=body,
        headers={"Authorization": f"Bearer {load_key()}",
                 "Content-Type": f"multipart/form-data; boundary={b}"})
    t0 = time.time()
    try:
        d = json.load(urllib.request.urlopen(req, timeout=600))
    except urllib.error.HTTPError as e:
        sys.exit(f"HTTP {e.code}: {e.read().decode()[:400]}")
    raw_out.write_bytes(base64.b64decode(d["data"][0]["b64_json"]))
    return time.time() - t0, d.get("usage", {}).get("output_tokens", 0)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("hero"); ap.add_argument("out"); ap.add_argument("prompt_file")
    ap.add_argument("--w", type=int, default=512); ap.add_argument("--h", type=int, default=896)
    ap.add_argument("--model", default="gpt-image-1-mini")
    ap.add_argument("--landscape", action="store_true", help="1536x1024 가로 생성")
    a = ap.parse_args()
    if a.landscape:
        globals()["GEN_W"], globals()["GEN_H"] = 1536, 1024
    out = pathlib.Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    raw = out.with_name(out.stem + "_raw.png")
    if out.exists():
        print(f"SKIP {out.name}"); sys.exit(0)
    dt, tok = edit(pathlib.Path(a.hero), pathlib.Path(a.prompt_file).read_text().strip(), a.model, raw)
    gw, gh = GEN_W, GEN_H
    cw = round(gh * a.w / a.h)
    if cw <= gw:
        vf = f"crop={cw}:{gh}:{(gw-cw)//2}:0,scale={a.w}:{a.h}:flags=lanczos"
    else:
        ch = round(gw * a.h / a.w)
        vf = f"crop={gw}:{ch}:0:{(gh-ch)//2},scale={a.w}:{a.h}:flags=lanczos"
    subprocess.run([FFMPEG, "-y", "-v", "error", "-i", str(raw), "-vf", vf, str(out)], check=True)
    print(f"{out.name}  {dt:.1f}초  출력토큰 {tok}")
    (out.parent / "images.tsv").open("a").write(f"{out.stem}\tedit\t{a.model}\t{tok}\t{dt:.1f}\n")
