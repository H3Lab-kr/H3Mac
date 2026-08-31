#!/usr/bin/env python3
"""h3 대화형 세션 드라이버 — 모델 로딩과 세션 캐시를 여러 컷에 분할상환한다.

h3 는 파이프에서 stdout 이 블록 버퍼링되어 'Done ->' 가 안 나온다. 그래서 pty 를 쓴다.

사용: lib/session.py <run-dir> <plan.tsv>
plan.tsv (탭 구분, 헤더 포함):
  라벨  프롬프트파일  첫프레임  가로  세로  스텝  레이어  프레임
"""
import os, sys, pty, re, csv, time, select, pathlib, signal, subprocess

ROOT = pathlib.Path(__file__).resolve().parent.parent
H3_DIR = ROOT / "engines" / "h3.c"
MODEL = os.environ.get("H3_MODEL", str(ROOT / "models" / "MiniMax-H3-turbo4"))
SEED = os.environ.get("SEED", "42")
ANSI = re.compile(rb"\x1b\[[0-9;?]*[A-Za-z]|\r")

class Session:
    def __init__(self, log):
        env = dict(os.environ)
        env["H3_FFMPEG"] = str(ROOT / "bin" / "ffmpeg")
        env["H3_FFPROBE"] = str(ROOT / "bin" / "ffprobe")
        env["TERM"] = "dumb"
        self.pid, self.fd = pty.fork()
        if self.pid == 0:                       # 자식
            os.chdir(H3_DIR)                    # 셰이더가 CWD 상대
            os.execve("./h3", ["./h3", "-d", MODEL], env)
        self.buf = b""
        self.log = open(log, "wb")

    def read_until(self, needle, timeout=3600):
        end = time.time() + timeout
        needle = needle.encode()
        while time.time() < end:
            r, _, _ = select.select([self.fd], [], [], 1.0)
            if r:
                try: chunk = os.read(self.fd, 65536)
                except OSError: break
                if not chunk: break
                self.log.write(chunk); self.log.flush()
                self.buf += chunk
            if needle in self.buf:
                i = self.buf.index(needle) + len(needle)
                out, self.buf = self.buf[:i], self.buf[i:]
                return ANSI.sub(b"", out).decode("utf-8", "replace")
        raise TimeoutError(f"'{needle.decode()}' 대기 시간 초과")

    def send(self, line):
        os.write(self.fd, (line + "\n").encode())

    def cmd(self, line):
        self.send(line); return self.read_until("h3> ")

    def close(self):
        try: self.send("!quit"); time.sleep(0.5)
        except Exception: pass
        try: os.kill(self.pid, signal.SIGTERM)
        except Exception: pass
        self.log.close()

def main():
    run = pathlib.Path(sys.argv[1]); plan = pathlib.Path(sys.argv[2])
    out = run / "out"; out.mkdir(parents=True, exist_ok=True)
    man = run / "manifest.tsv"
    if not man.exists():
        man.write_text("라벨\t가로\t세로\t프레임\t스텝\t레이어\t시드\t초\t추가옵션\t첫프레임\t파일\n")

    rows = [r for r in csv.DictReader(plan.open(), delimiter="\t")]
    rows = [r for r in rows if not (out / f"{r['라벨']}.mp4").exists()]
    if not rows: print("전부 존재함 — 할 일 없음"); return

    t_start = time.time()
    s = Session(run / "session.log")
    banner = s.read_until("h3> ", timeout=1800)
    t_load = time.time() - t_start
    print(f"세션 기동 {t_load:.0f}초", flush=True)
    s.cmd(f"!output {out.resolve()}")
    s.cmd(f"!seed {SEED}")

    for r in rows:
        lbl = r["라벨"]
        s.cmd(f"!size {r['가로']}x{r['세로']}")
        s.cmd(f"!frames {r['프레임']}")
        s.cmd(f"!steps {r['스텝']}")
        s.cmd(f"!layers {r['레이어']}")
        ff = r.get("첫프레임", "").strip()
        s.cmd(f"!first {ff}" if ff else "!first clear")
        prompt = " ".join(pathlib.Path(r["프롬프트파일"]).read_text().split())
        t0 = time.time()
        s.send(prompt)
        txt = s.read_until("Done -> ")
        tail = s.read_until("h3> ")
        dt = time.time() - t0
        m = re.search(r"\[([0-9.]+)s\]", tail)
        inner = float(m.group(1)) if m else float("nan")
        s.cmd(f"!save {(out/(lbl+'.mp4')).resolve()}")
        with man.open("a") as f:
            f.write(f"{lbl}\t{r['가로']}\t{r['세로']}\t{r['프레임']}\t{r['스텝']}\t"
                    f"{r['레이어']}\t{SEED}\t{dt:.0f}\t세션\t{pathlib.Path(ff).name if ff else ''}\tout/{lbl}.mp4\n")
        print(f"DONE {lbl} {dt:.0f}초 (h3 내부 {inner:.1f}초)", flush=True)
    s.close()
    print(f"SESSION-DONE 총 {time.time()-t_start:.0f}초 (기동 {t_load:.0f}초 포함)")

if __name__ == "__main__": main()
