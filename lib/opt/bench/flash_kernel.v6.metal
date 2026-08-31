#include <metal_stdlib>
#include <metal_simdgroup_matrix>
using namespace metal;

/* ── 윈도우 FlashAttention v3 ─────────────────────────────────────────
 * v2 대비 핵심 변경: O 누산기를 threadgroup 메모리 → 레지스터(simdgroup 행렬).
 * 행별 재스케일은 대각행렬 곱으로 처리한다 (diag(alpha)·O).
 *   v2 는 키 타일마다 O 16장을 threadgroup 에서 load+store 했다 — 4KB 왕복 × 타일수.
 *   v3 는 그 왕복을 없애고 MMA 16회(전체의 6%)로 대체한다.
 * 소프트맥스도 8레인 → 32레인(행당 4레인)으로 분산한다. */

struct flash_args {
    uint heads, rows, sink, w0, w1, q0, q_count;
    float scale;
};

#define FA_D   128
#define FA_QS    8
#define FA_SG    6
#define FA_KT   64
#define FA_Q  (FA_QS * FA_SG)
#define FA_DT (FA_D / 8)
#define FA_JT (FA_KT / 8)

kernel void h3_flash_window_bf16(
    device const ushort *Q [[buffer(0)]],
    device const ushort *K [[buffer(1)]],
    device const ushort *V [[buffer(2)]],
    device ushort *O [[buffer(3)]],
    constant flash_args &a [[buffer(4)]],
    threadgroup char *shared [[threadgroup(0)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint tid [[thread_index_in_threadgroup]]) {

    const uint head = tgid.y;
    const uint qend = a.q0 + a.q_count;
    const uint qbase = a.q0 + tgid.x * FA_Q + sg * FA_QS;
    const uint qload = qbase + FA_QS <= qend ? qbase :
                       (qend >= FA_QS ? qend - FA_QS : 0);

    threadgroup bfloat *kt = (threadgroup bfloat *)shared;      /* [KT][D] */
    threadgroup bfloat *vt = kt + FA_KT * FA_D;
    threadgroup bfloat *pt = vt + FA_KT * FA_D;                 /* [SG][QS][KT] */
    threadgroup float  *sft = (threadgroup float *)(pt + FA_SG * FA_QS * FA_KT);
    threadgroup float  *dg = sft + FA_SG * FA_QS * FA_KT;       /* [SG][8][8] 대각 */
    threadgroup float  *ms = dg + FA_SG * 64;
    threadgroup float  *ls = ms + FA_SG * FA_QS;

    threadgroup bfloat *P = pt + sg * FA_QS * FA_KT;
    threadgroup float  *S = sft + sg * FA_QS * FA_KT;
    threadgroup bfloat *Sb = (threadgroup bfloat *)(sft) + sg * FA_QS * FA_KT;
    threadgroup float  *Dg = dg + sg * 64;

    const device bfloat *Qh = (const device bfloat *)Q + (size_t)head * a.rows * FA_D;
    const device bfloat *Kh = (const device bfloat *)K + (size_t)head * a.rows * FA_D;
    const device bfloat *Vh = (const device bfloat *)V + (size_t)head * a.rows * FA_D;
    device bfloat *Oh = (device bfloat *)O + (size_t)head * a.rows * FA_D;

    if (lane < FA_QS) { ms[sg * FA_QS + lane] = -INFINITY; ls[sg * FA_QS + lane] = 0.f; }
    for (uint i = lane; i < 64; i += 32) Dg[i] = 0.f;

    simdgroup_bfloat8x8 Qm[FA_DT];
    for (uint d = 0; d < FA_DT; d++)
        simdgroup_load(Qm[d], Qh + (size_t)qload * FA_D + d * 8, FA_D);
    simdgroup_float8x8 Om[FA_DT];
    for (uint d = 0; d < FA_DT; d++)
        Om[d] = make_filled_simdgroup_matrix<float, 8, 8>(0.f);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint seg = 0; seg < 2; seg++) {
        uint s0 = seg == 0 ? 0u : a.w0;
        uint s1 = seg == 0 ? a.sink : a.w1;
        for (uint kb = s0; kb < s1; kb += FA_KT) {
            uint klen = min((uint)FA_KT, s1 - kb);
            for (uint i = tid * 4; i < FA_KT * FA_D; i += FA_SG * 32 * 4) {
                uint r = i / FA_D, c = i % FA_D;
                if (r < klen) {
                    const device bfloat *kp = Kh + (size_t)(kb + r) * FA_D + c;
                    const device bfloat *vp = Vh + (size_t)(kb + r) * FA_D + c;
                    kt[i]=kp[0]; kt[i+1]=kp[1]; kt[i+2]=kp[2]; kt[i+3]=kp[3];
                    vt[i]=vp[0]; vt[i+1]=vp[1]; vt[i+2]=vp[2]; vt[i+3]=vp[3];
                } else {
                    kt[i]=kt[i+1]=kt[i+2]=kt[i+3]=bfloat(0);
                    vt[i]=vt[i+1]=vt[i+2]=vt[i+3]=bfloat(0);
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            for (uint j = 0; j < FA_JT; j++) {
                simdgroup_bfloat8x8 acc = simdgroup_bfloat8x8(0);
                for (uint d = 0; d < FA_DT; d++) {
                    simdgroup_bfloat8x8 Km;
                    simdgroup_load(Km, kt + j * 8 * FA_D + d * 8, FA_D, 0, true);
                    simdgroup_multiply_accumulate(acc, Qm[d], Km, acc);
                }
                simdgroup_store(acc, Sb + j * 8, FA_KT);
            }
            simdgroup_barrier(mem_flags::mem_threadgroup);

            /* 소프트맥스: 행당 4레인 (8행 × 4 = 32레인 전부 사용) */
            {
                uint row = lane / 4, part = lane % 4;
                threadgroup bfloat *r = Sb + row * FA_KT;
                float lm = -INFINITY;
                for (uint j = part; j < FA_KT; j += 4)
                    if (j < klen) lm = max(lm, float(r[j]) * a.scale);
                for (uint o = 1; o < 4; o <<= 1)
                    lm = max(lm, simd_shuffle_xor(lm, o));
                float m_old = ms[sg * FA_QS + row];
                float m_new = max(m_old, lm);
                float sum = 0.f;
                for (uint j = part; j < FA_KT; j += 4) {
                    float p = j < klen ? exp(float(r[j]) * a.scale - m_new) : 0.f;
                    P[row * FA_KT + j] = bfloat(p);
                    sum += p;
                }
                for (uint o = 1; o < 4; o <<= 1)
                    sum += simd_shuffle_xor(sum, o);
                if (part == 0) {
                    float alpha = isinf(m_old) ? 0.f : exp(m_old - m_new);
                    ms[sg * FA_QS + row] = m_new;
                    ls[sg * FA_QS + row] = ls[sg * FA_QS + row] * alpha + sum;
                    Dg[row * 8 + row] = alpha;
                }
            }
            simdgroup_barrier(mem_flags::mem_threadgroup);

            /* O ← diag(alpha)·O  (레지스터 유지, threadgroup 왕복 없음) */
            simdgroup_float8x8 Dm;
            simdgroup_load(Dm, Dg, 8);
            for (uint d = 0; d < FA_DT; d++) {
                simdgroup_float8x8 t;
                simdgroup_multiply(t, Dm, Om[d]);
                Om[d] = t;
            }
            /* O += P·V */
            for (uint d = 0; d < FA_DT; d++) {
                simdgroup_float8x8 acc = Om[d];
                for (uint j = 0; j < FA_JT; j++) {
                    simdgroup_bfloat8x8 Pm, Vm;
                    simdgroup_load(Pm, P + j * 8, FA_KT);
                    simdgroup_load(Vm, vt + j * 8 * FA_D + d * 8, FA_D);
                    simdgroup_multiply_accumulate(acc, Pm, Vm, acc);
                }
                Om[d] = acc;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }
    /* 정규화 후 출력. K/V 타일 공간을 스테이징으로 재사용한다
     * (SG 4 × 8행 × 128 = 4096 float = 16KB ≤ kt+vt 32KB). */
    threadgroup_barrier(mem_flags::mem_threadgroup);
    threadgroup float *Ost = (threadgroup float *)kt + sg * FA_QS * FA_D;
    for (uint d = 0; d < FA_DT; d++)
        simdgroup_store(Om[d], Ost + d * 8, FA_D);
    simdgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i = lane; i < FA_QS * FA_D; i += 32) {
        uint row = qload + i / FA_D;
        if (row >= a.q0 && row < qend) {
            float l = ls[sg * FA_QS + i / FA_D];
            Oh[(size_t)row * FA_D + i % FA_D] = bfloat(l > 0.f ? Ost[i] / l : 0.f);
        }
    }
}
