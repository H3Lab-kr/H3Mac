#include <metal_stdlib>
#include <metal_simdgroup_matrix>
using namespace metal;

/* ── 윈도우 FlashAttention (bf16 · simdgroup 행렬) ────────────────────
 * 세 가지를 한 번에 해결한다:
 *   1. 개더/스캐터 제거 — 원본 Q/K/V 를 직접 읽는다 (스테이징 복사 0)
 *   2. 싱크 중복 제거 — [0,sink) 와 [w0,w1) 두 구간을 이어서 순회하고
 *      온라인 소프트맥스로 병합한다 (log-sum-exp 문제 해결)
 *   3. 점수 행렬 비실체화
 * 레이아웃: Q/K/V/O 헤드 우선 [H][rows][128] bf16.
 * O 는 threadgroup 메모리에 두어 행별 재스케일이 가능하게 한다. */

struct flash_args {
    uint heads, rows, sink, w0, w1, q0, q_count;
    float scale;
};

#define FA_D   128
#define FA_QS    8       /* simdgroup 당 쿼리 행 */
#define FA_SG    4       /* threadgroup 당 simdgroup */
#define FA_KT   64       /* 키 타일 */
#define FA_Q  (FA_QS * FA_SG)

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
    const uint qbase = a.q0 + tgid.x * FA_Q + sg * FA_QS;
    const uint qend = a.q0 + a.q_count;

    threadgroup bfloat *kt = (threadgroup bfloat *)shared;          /* [KT][D] */
    threadgroup bfloat *vt = kt + FA_KT * FA_D;
    threadgroup bfloat *pt = vt + FA_KT * FA_D;                     /* [SG][QS][KT] bf16 */
    threadgroup float  *sft = (threadgroup float *)(pt + FA_SG * FA_QS * FA_KT);
    threadgroup float  *ot = sft + FA_SG * FA_QS * FA_KT;           /* 점수는 별도 버퍼 */
    threadgroup float  *rs = ot + FA_SG * FA_QS * FA_D;             /* [SG][QS] alpha */
    threadgroup float  *ms = rs + FA_SG * FA_QS;
    threadgroup float  *ls = ms + FA_SG * FA_QS;

    threadgroup bfloat *P = pt + sg * FA_QS * FA_KT;
    threadgroup float  *S = sft + sg * FA_QS * FA_KT;
    threadgroup float  *Ot = ot + sg * FA_QS * FA_D;

    const device bfloat *Qh = (const device bfloat *)Q + (size_t)head * a.rows * FA_D;
    const device bfloat *Kh = (const device bfloat *)K + (size_t)head * a.rows * FA_D;
    const device bfloat *Vh = (const device bfloat *)V + (size_t)head * a.rows * FA_D;
    device bfloat *Oh = (device bfloat *)O + (size_t)head * a.rows * FA_D;

    for (uint i = lane; i < FA_QS * FA_D; i += 32) Ot[i] = 0.f;
    if (lane < FA_QS) { ms[sg * FA_QS + lane] = -INFINITY; ls[sg * FA_QS + lane] = 0.f; }

    /* 꼬리 처리: 범위를 넘으면 마지막 유효 타일을 읽어 계산만 하고 저장하지 않는다.
     * (배리어가 갈라지면 안 되므로 조기 반환하지 않는다) */
    const uint qload = qbase + FA_QS <= qend ? qbase :
                       (qend >= FA_QS ? qend - FA_QS : 0);
    simdgroup_bfloat8x8 Qm[FA_D / 8];
    for (uint d = 0; d < FA_D / 8; d++)
        simdgroup_load(Qm[d], Qh + (size_t)qload * FA_D + d * 8, FA_D);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint seg = 0; seg < 2; seg++) {
        uint s0 = seg == 0 ? 0u : a.w0;
        uint s1 = seg == 0 ? a.sink : a.w1;
        for (uint kb = s0; kb < s1; kb += FA_KT) {
            uint klen = min((uint)FA_KT, s1 - kb);
            /* 4폭 벡터 적재 */
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
            /* S = Q·Kᵀ → P (bfloat 로 저장) */
            for (uint j = 0; j < FA_KT / 8; j++) {
                simdgroup_float8x8 acc = make_filled_simdgroup_matrix<float, 8, 8>(0.f);
                for (uint d = 0; d < FA_D / 8; d++) {
                    simdgroup_bfloat8x8 Km;
                    simdgroup_load(Km, kt + j * 8 * FA_D + d * 8, FA_D, 0, true);
                    simdgroup_multiply_accumulate(acc, Qm[d], Km, acc);
                }
                simdgroup_store(acc, S + j * 8, FA_KT);
            }
            simdgroup_barrier(mem_flags::mem_threadgroup);

            /* 온라인 소프트맥스: lane 하나가 쿼리 한 행 */
            if (lane < FA_QS) {
                threadgroup float *row = S + lane * FA_KT;
                float m_old = ms[sg * FA_QS + lane], m_new = m_old;
                for (uint j = 0; j < klen; j++) m_new = max(m_new, row[j] * a.scale);
                float alpha = isinf(m_old) ? 0.f : exp(m_old - m_new);
                float sum = 0.f;
                for (uint j = 0; j < FA_KT; j++) {
                    float p = j < klen ? exp(row[j] * a.scale - m_new) : 0.f;
                    row[j] = p; sum += p;
                }
                ms[sg * FA_QS + lane] = m_new;
                ls[sg * FA_QS + lane] = ls[sg * FA_QS + lane] * alpha + sum;
                rs[sg * FA_QS + lane] = alpha;
            }
            simdgroup_barrier(mem_flags::mem_threadgroup);
            for (uint i = lane; i < FA_QS * FA_KT; i += 32) P[i] = bfloat(S[i]);
            simdgroup_barrier(mem_flags::mem_threadgroup);

            /* O 행별 재스케일 후 P·V 누적 */
            for (uint i = lane; i < FA_QS * FA_D; i += 32)
                Ot[i] *= rs[sg * FA_QS + i / FA_D];
            simdgroup_barrier(mem_flags::mem_threadgroup);
            for (uint d = 0; d < FA_D / 8; d++) {
                simdgroup_float8x8 acc;
                simdgroup_load(acc, Ot + d * 8, FA_D);
                for (uint j = 0; j < FA_KT / 8; j++) {
                    simdgroup_bfloat8x8 Pm, Vm;
                    simdgroup_load(Pm, P + j * 8, FA_KT);
                    simdgroup_load(Vm, vt + j * 8 * FA_D + d * 8, FA_D);
                    simdgroup_multiply_accumulate(acc, Pm, Vm, acc);
                }
                simdgroup_store(acc, Ot + d * 8, FA_D);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }
    for (uint i = lane; i < FA_QS * FA_D; i += 32) {
        uint row = qload + i / FA_D;
        if (row >= a.q0 && row < qend) {
            float l = ls[sg * FA_QS + i / FA_D];
            Oh[(size_t)row * FA_D + i % FA_D] = bfloat(l > 0.f ? Ot[i] / l : 0.f);
        }
    }
}
