#!/usr/bin/env python3
"""프로젝트별 통합 프롬프트 기록을 생성한다.

stills / motion / chars / music 프롬프트는 각각이 원본이다. 이 도구는 그것들을
샷 단위로 합치고 실행 결과(manifest.tsv)를 붙여 하나의 문서로 만든다.

**생성물이다. 손으로 고치지 않는다.** 원본을 고치고 다시 돌린다.
사용: lib/promptdoc.py <프로젝트> [실행디렉터리...]
"""
import sys, csv, pathlib, textwrap

ROOT = pathlib.Path(__file__).resolve().parent.parent
P = ROOT / "prompts"

def read(p):
    return " ".join(p.read_text().split()) if p.exists() else None

def field(txt, name):
    """3필드/6섹션 프롬프트에서 한 필드만 뽑는다."""
    if not txt or f"{name}:" not in txt: return None
    rest = txt.split(f"{name}:", 1)[1]
    for nxt in ("integrated_multimodal_description:", "overall_soundscape:", "non_diegetic_music:"):
        if nxt in rest: rest = rest.split(nxt, 1)[0]
    return rest.strip()

def load_manifests(runs):
    rows = {}
    for r in runs:
        man = pathlib.Path(r) / "manifest.tsv"
        if not man.exists(): continue
        for d in csv.DictReader(man.open(), delimiter="\t"):
            rows[d["라벨"]] = (pathlib.Path(r).name, d)
    return rows

def main():
    proj = sys.argv[1]
    runs = sys.argv[2:] or sorted(str(p) for p in (ROOT/"runs").glob(f"*-{proj}*"))
    d = P / proj
    if not d.exists(): sys.exit(f"없는 프로젝트: {proj}")
    man = load_manifests(runs)

    stills = {p.stem: p for p in sorted((d/"stills").glob("*.txt"))}
    motion = {p.stem: p for p in sorted((d/"motion").glob("*.txt"))}
    chars  = {p.stem: p for p in sorted((d/"chars").glob("*.txt"))}
    music  = {p.stem: p for p in sorted((d/"music").glob("*.txt"))}
    labels = sorted(set(stills) | set(motion))

    o = [f"# {proj} — 통합 프롬프트 기록", "",
         "> `lib/promptdoc.py` 가 생성한다. **손으로 고치지 않는다.**",
         "> 원본은 `stills/` `motion/` `chars/` `music/` 안에 있다.", ""]
    if runs:
        o += ["**실행 디렉터리**: " + " · ".join(f"`{pathlib.Path(r).name}`" for r in runs), ""]

    if chars:
        o += ["## 캐릭터 시트", ""]
        for k, p in chars.items():
            o += [f"### {k}", "", "```", textwrap.fill(read(p), 96), "```", ""]

    o += ["## 샷", ""]
    for k in labels:
        m = man.get(k)
        o.append(f"### {k}")
        if m:
            run, r = m
            o += ["", f"| 실행 | 캔버스 | 프레임 | 스텝 | 레이어 | 시드 | 렌더 | 첫 프레임 |",
                  "|---|---|---:|---:|---:|---:|---:|---|",
                  f"| `{run}` | {r['가로']}×{r['세로']} | {r['프레임']} | {r['스텝']} | "
                  f"{r['레이어']} | {r['시드']} | {r['초']}초 | {r.get('첫프레임','') or '—'} |"]
        o.append("")
        if k in stills:
            o += ["**스틸 프롬프트**", "", "```", textwrap.fill(read(stills[k]), 96), "```", ""]
        if k in motion:
            t = read(motion[k])
            for name, title in (("integrated_multimodal_description", "동작·화면"),
                                ("overall_soundscape", "사운드스케이프"),
                                ("non_diegetic_music", "음악")):
                v = field(t, name)
                if v and v != "N/A":
                    o += [f"**{title}**", "", "```", textwrap.fill(v, 96), "```", ""]
                elif v == "N/A":
                    o += [f"**{title}** — 없음", ""]

    if music:
        o += ["## 음악 베드", ""]
        for k, p in music.items():
            v = field(read(p), "non_diegetic_music") or read(p)
            m = man.get(k)
            if m: o += [f"### {k} — {m[1]['프레임']}프레임 · {m[1]['초']}초", ""]
            else: o += [f"### {k}", ""]
            o += ["```", textwrap.fill(v, 96), "```", ""]

    out = d / "SHOTS.md"
    out.write_text("\n".join(o))
    print(f"  {out.relative_to(ROOT)}  샷 {len(labels)}개 · 실행 기록 {len(man)}건")

if __name__ == "__main__": main()
