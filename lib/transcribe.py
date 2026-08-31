#!/usr/bin/env python3
"""영상의 한국어 대사를 전사한다. 발음이 살아있는지 판정하는 계측기.

로컬 whisper 대신 OpenAI 전사 API 를 쓴다 (torch 설치 회피).
사용: lib/transcribe.py <video.mp4> [--model whisper-1]
"""
import os, sys, json, uuid, subprocess, urllib.request, pathlib, argparse, tempfile

ROOT = pathlib.Path(__file__).resolve().parent.parent

def load_key():
    if key := os.environ.get("OPENAI_API_KEY"):
        return key.strip()
    env_file = ROOT / ".env"
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            if line.startswith("OPENAI_API_KEY"):
                return line.split("=", 1)[1].strip().strip('"').strip("'")
    raise SystemExit("OPENAI_API_KEY 환경변수 또는 .env 파일이 필요합니다.")

def extract(video, wav):
    subprocess.run([str(ROOT/"bin"/"ffmpeg"), "-y", "-v", "error", "-i", str(video),
                    "-vn", "-ac", "1", "-ar", "16000", str(wav)], check=True)

def transcribe(wav, model, lang="ko"):
    b = uuid.uuid4().hex
    def part(n, v):
        return f'--{b}\r\nContent-Disposition: form-data; name="{n}"\r\n\r\n{v}\r\n'.encode()
    body = part("model", model) + part("language", lang)
    if model == "whisper-1":
        body += part("response_format", "verbose_json")
    body += (f'--{b}\r\nContent-Disposition: form-data; name="file"; filename="a.wav"\r\n'
             f'Content-Type: audio/wav\r\n\r\n').encode() + wav.read_bytes() + b"\r\n"
    body += f"--{b}--\r\n".encode()
    req = urllib.request.Request("https://api.openai.com/v1/audio/transcriptions", data=body,
        headers={"Authorization": f"Bearer {load_key()}",
                 "Content-Type": f"multipart/form-data; boundary={b}"})
    try:
        return json.load(urllib.request.urlopen(req, timeout=180))
    except urllib.error.HTTPError as e:
        return {"error": f"HTTP {e.code}: {e.read().decode()[:200]}"}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("video"); ap.add_argument("--model", default="whisper-1")
    a = ap.parse_args()
    with tempfile.TemporaryDirectory() as d:
        w = pathlib.Path(d) / "a.wav"
        extract(a.video, w)
        r = transcribe(w, a.model)
    name = pathlib.Path(a.video).stem
    if "error" in r:
        print(f"{name}\t{r['error']}"); sys.exit(1)
    text = (r.get("text") or "").strip()
    segs = r.get("segments") or []
    # avg_logprob 로 실제 발화와 환각을 가른다 (실제 −0.2~−0.4 · 환각 −4~−6)
    lp = sum(s.get("avg_logprob", 0) for s in segs) / len(segs) if segs else float("nan")
    nsp = sum(s.get("no_speech_prob", 0) for s in segs) / len(segs) if segs else float("nan")
    print(f"{name}\tlogprob {lp:+.2f}\t무발화확률 {nsp:.2f}\t「{text}」")
