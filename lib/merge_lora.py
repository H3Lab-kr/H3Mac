#!/usr/bin/env python3
"""turbo LoRA 를 FL2VA transformer 에 오프라인 병합한다.

W' = W + (alpha/rank) * (B @ A)

numpy 에 bfloat16 이 없으므로 safetensors 를 직접 파싱하고 uint16 비트로 다룬다.
fp32 왕복은 round-to-nearest-even 으로 되돌린다.
"""
import json, sys, struct, pathlib, argparse, time
import numpy as np

def read_st(path):
    """(헤더dict, mmap) 반환. 데이터는 헤더 끝 오프셋 기준."""
    f = open(path, "rb")
    n = struct.unpack("<Q", f.read(8))[0]
    hdr = json.loads(f.read(n))
    base = 8 + n
    mm = np.memmap(path, dtype=np.uint8, mode="r")
    return hdr, mm, base

DT = {"BF16": (np.uint16, 2), "F32": (np.float32, 4), "F16": (np.float16, 2),
      "I64": (np.int64, 8), "I32": (np.int32, 4), "U8": (np.uint8, 1), "BOOL": (np.uint8, 1)}

def tensor(hdr, mm, base, name):
    m = hdr[name]
    s, e = m["data_offsets"]
    dt, _ = DT[m["dtype"]]
    return np.frombuffer(mm[base+s:base+e].tobytes(), dtype=dt).reshape(m["shape"])

def bf16_to_f32(u):
    return (u.astype(np.uint32) << 16).view(np.float32)

def f32_to_bf16(f):
    u = f.view(np.uint32)
    u = (u + 0x7FFF + ((u >> 16) & 1)) >> 16      # round-to-nearest-even
    return u.astype(np.uint16)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True); ap.add_argument("--dst", required=True)
    ap.add_argument("--lora", required=True); ap.add_argument("--scale", type=float, default=1.0)
    a = ap.parse_args()
    src, dst = pathlib.Path(a.src), pathlib.Path(a.dst)
    dst.mkdir(parents=True, exist_ok=True)

    lh, lm, lb = read_st(a.lora)
    mods = sorted({k.rsplit(".lora_", 1)[0].rsplit(".alpha", 1)[0]
                   for k in lh if k != "__metadata__"})
    print(f"LoRA 모듈 {len(mods)}개")

    idx = json.load(open(src / "model.safetensors.index.json"))
    wm = idx["weight_map"]
    # diffusion_model.X -> X.weight
    plan = {}
    for m in mods:
        t = m.replace("diffusion_model.", "", 1) + ".weight"
        if t in wm: plan.setdefault(wm[t], {})[t] = m
        else: print("  ! 모델에 없음:", t)
    print(f"대상 텐서 {sum(len(v) for v in plan.values())}개 · 샤드 {len(plan)}개")

    for shard in sorted(set(wm.values())):
        out = dst / shard
        if out.exists(): print(f"SKIP {shard}"); continue
        t0 = time.time()
        h, mmp, b = read_st(src / shard)
        names = [k for k in h if k != "__metadata__"]
        todo = plan.get(shard, {})
        blobs, nh, off = [], {}, 0
        for nm in names:
            meta = h[nm]
            if nm in todo:
                mod = todo[nm]
                W = bf16_to_f32(tensor(h, mmp, b, nm)).astype(np.float32)
                A = bf16_to_f32(tensor(lh, lm, lb, mod + ".lora_A.weight")).astype(np.float32)
                B = bf16_to_f32(tensor(lh, lm, lb, mod + ".lora_B.weight")).astype(np.float32)
                al = float(tensor(lh, lm, lb, mod + ".alpha")) if mod + ".alpha" in lh else A.shape[0]
                W += (a.scale * al / A.shape[0]) * (B @ A)
                raw = f32_to_bf16(W).tobytes()
            else:
                s, e = meta["data_offsets"]
                raw = mmp[b+s:b+e].tobytes()
            blobs.append(raw)
            nh[nm] = {"dtype": meta["dtype"], "shape": meta["shape"],
                      "data_offsets": [off, off + len(raw)]}
            off += len(raw)
        hb = json.dumps(nh, separators=(",", ":")).encode()
        hb += b" " * ((8 - len(hb) % 8) % 8)
        with open(out, "wb") as f:
            f.write(struct.pack("<Q", len(hb))); f.write(hb)
            for r in blobs: f.write(r)
        print(f"DONE {shard}  병합 {len(todo)}개  {time.time()-t0:.0f}초")
    print("MERGE-DONE")

if __name__ == "__main__": main()
