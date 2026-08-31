#include <metal_stdlib>
#include <metal_simdgroup_matrix>
using namespace metal;

/* ── H3Lab 고속 윈도우 FlashAttention-2 커널 ─────────────────────────
 * 최적화 요약:
 * 1. O 누산기를 레지스터(simdgroup_float8x8)에 상주
 * 2. diag(alpha) 행렬 재스케일로 Threadgroup 메모리 왕복 제거
 * 3. Softmax: 행당 4레인 병렬 축약 (SIMD Shuffle)
 * 4. QK 내적: FP32 누산기(simdgroup_float8x8)로 정밀도 완벽 보존
 * 5. K/V 타일 로드: 4배 언롤링 및 대역폭 최적화
 * 6. 총 Threadgroup 메모리: 27,520 바이트 (< 32,768 바이트 하드웨어 한도 내 안전)
 */

struct flash_args {
    uint heads, rows, sink, w0, w1, q0, q_count;
    float scale;
};

#define FA_D   128
#define FA_QS    8
#define FA_SG    8
#define FA_KT   32
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

    threadgroup bfloat *kt = (threadgroup bfloat *)shared;                       /* [32][128] = 8192 B */
    threadgroup bfloat *vt = kt + FA_KT * FA_D;                                  /* [32][128] = 8192 B */
    threadgroup bfloat *pt = vt + FA_KT * FA_D;                                  /* [SG][8][32] = 3072 B */
    threadgroup float  *sft = (threadgroup float *)(pt + FA_SG * FA_QS * FA_KT); /* [SG][8][32] = 6144 B */
    threadgroup float  *dg = sft + FA_SG * FA_QS * FA_KT;                        /* [SG][64] = 1536 B */
    threadgroup float  *ms = dg + FA_SG * 64;                                    /* [SG][8] = 192 B */
    threadgroup float  *ls = ms + FA_SG * FA_QS;                                 /* [SG][8] = 192 B */

    threadgroup bfloat *P = pt + sg * FA_QS * FA_KT;
    threadgroup float  *S = sft + sg * FA_QS * FA_KT;
    threadgroup float  *Dg = dg + sg * 64;

    const device bfloat *Qh = (const device bfloat *)Q + (size_t)head * a.rows * FA_D;
    const device bfloat *Kh = (const device bfloat *)K + (size_t)head * a.rows * FA_D;
    const device bfloat *Vh = (const device bfloat *)V + (size_t)head * a.rows * FA_D;
    device bfloat *Oh = (device bfloat *)O + (size_t)head * a.rows * FA_D;

    if (lane < FA_QS) {
        ms[sg * FA_QS + lane] = -INFINITY;
        ls[sg * FA_QS + lane] = 0.f;
    }
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
                    kt[i]   = kp[0]; kt[i+1] = kp[1]; kt[i+2] = kp[2]; kt[i+3] = kp[3];
                    vt[i]   = vp[0]; vt[i+1] = vp[1]; vt[i+2] = vp[2]; vt[i+3] = vp[3];
                } else {
                    kt[i] = kt[i+1] = kt[i+2] = kt[i+3] = bfloat(0);
                    vt[i] = vt[i+1] = vt[i+2] = vt[i+3] = bfloat(0);
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            /* 1) Q · K^T 내적 (FP32 정확도 누산) */
            for (uint j = 0; j < FA_JT; j++) {
                simdgroup_float8x8 acc = make_filled_simdgroup_matrix<float, 8, 8>(0.f);
                for (uint d = 0; d < FA_DT; d++) {
                    simdgroup_bfloat8x8 Km;
                    simdgroup_load(Km, kt + j * 8 * FA_D + d * 8, FA_D, 0, true);
                    simdgroup_multiply_accumulate(acc, Qm[d], Km, acc);
                }
                simdgroup_store(acc, S + j * 8, FA_KT);
            }
            simdgroup_barrier(mem_flags::mem_threadgroup);

            /* 2) 온라인 소프트맥스 (행당 4개 레인 병렬 리덕션) */
            {
                uint row = lane / 4, part = lane % 4;
                threadgroup float *r = S + row * FA_KT;
                float lm = -INFINITY;
                for (uint j = part; j < FA_KT; j += 4)
                    if (j < klen) lm = max(lm, r[j] * a.scale);
                for (uint o = 1; o < 4; o <<= 1)
                    lm = max(lm, simd_shuffle_xor(lm, o));
                float m_old = ms[sg * FA_QS + row];
                float m_new = max(m_old, lm);
                float sum = 0.f;
                for (uint j = part; j < FA_KT; j += 4) {
                    float p = j < klen ? exp(r[j] * a.scale - m_new) : 0.f;
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

            /* 3) O ← diag(alpha) · O (레지스터 내부 재스케일) */
            simdgroup_float8x8 Dm;
            simdgroup_load(Dm, Dg, 8);
            for (uint d = 0; d < FA_DT; d++) {
                simdgroup_float8x8 t;
                simdgroup_multiply(t, Dm, Om[d]);
                Om[d] = t;
            }

            /* 4) O += P · V (P 1회 로드 공유) */
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

    /* 5) 정규화 및 최종 출력 저장 (Dg 스크래치 재사용으로 스테이징 메모리 절약) */
    threadgroup float *Ost = Dg; /* [64] 재사용 */
    for (uint d = 0; d < FA_DT; d++) {
        simdgroup_barrier(mem_flags::mem_threadgroup);
        simdgroup_store(Om[d], Ost, 8);
        simdgroup_barrier(mem_flags::mem_threadgroup);
        for (uint i = lane; i < 64; i += 32) {
            uint rr = i / 8, cc = i % 8;
            uint row = qload + rr;
            if (row >= a.q0 && row < qend && (qbase + rr) < qend) {
                float l = ls[sg * FA_QS + rr];
                Oh[(size_t)row * FA_D + d * 8 + cc] = bfloat(l > 0.f ? Ost[i] / l : 0.f);
            }
        }
    }
}
