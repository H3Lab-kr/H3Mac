#include <metal_stdlib>
#include <metal_simdgroup_matrix>
using namespace metal;

/* ── H3Lab INT8 vs INT4 Group-128 양자화 벤치마크 셰이더 ─────────────────
 * 1. INT8: 1바이트 1가중치, per-row scale
 * 2. INT4: 1바이트 2가중치(nibble), Group-128 scale + zero-point
 */

#define TILE_M 8
#define TILE_N 8
#define TILE_K 8

// 1. INT8 GEMV / Linear: 8-bit 로드 후 BF16 변환
kernel void h3_int8_linear_bench(
    device const char *W_i8 [[buffer(0)]],       // [N, K]
    device const float *scales [[buffer(1)]],     // [N]
    device const bfloat *X [[buffer(2)]],         // [M, K]
    device bfloat *Y [[buffer(3)]],               // [M, N]
    constant uint3 &dims [[buffer(4)]],           // M, N, K
    uint2 gid [[thread_position_in_grid]]) {
    
    uint m = gid.x;
    uint n = gid.y;
    if (m >= dims.x || n >= dims.y) return;
    
    float acc = 0.0f;
    uint K = dims.z;
    device const char *w_row = W_i8 + n * K;
    device const bfloat *x_row = X + m * K;
    float scale = scales[n];
    
    for (uint k = 0; k < K; k += 4) {
        float4 w = float4(w_row[k], w_row[k+1], w_row[k+2], w_row[k+3]) * scale;
        float4 x = float4(float(x_row[k]), float(x_row[k+1]), float(x_row[k+2]), float(x_row[k+3]));
        acc += dot(w, x);
    }
    
    Y[m * dims.y + n] = bfloat(acc);
}

// 2. INT4 Group-128 GEMV: 4-bit packed 로드 후 비트 언팩 및 Group Scale 적용
kernel void h3_int4_linear_bench(
    device const uchar *W_i4 [[buffer(0)]],      // [N, K / 2]
    device const float *scales [[buffer(1)]],     // [N, K / 128]
    device const bfloat *X [[buffer(2)]],         // [M, K]
    device bfloat *Y [[buffer(3)]],               // [M, N]
    constant uint3 &dims [[buffer(4)]],           // M, N, K
    uint2 gid [[thread_position_in_grid]]) {
    
    uint m = gid.x;
    uint n = gid.y;
    if (m >= dims.x || n >= dims.y) return;
    
    float acc = 0.0f;
    uint K = dims.z;
    uint K_half = K / 2;
    device const uchar *w_row = W_i4 + n * K_half;
    device const float *scale_row = scales + n * (K / 128);
    device const bfloat *x_row = X + m * K;
    
    for (uint k = 0; k < K_half; k += 2) {
        uchar b0 = w_row[k];
        uchar b1 = w_row[k+1];
        
        // 4비트 언팩 (nibble to signed int -8 ~ +7)
        int w0 = int(b0 & 0x0Fu) - 8;
        int w1 = int(b0 >> 4) - 8;
        int w2 = int(b1 & 0x0Fu) - 8;
        int w3 = int(b1 >> 4) - 8;
        
        float s = scale_row[(k * 2) / 128];
        float4 w = float4(w0, w1, w2, w3) * s;
        float4 x = float4(float(x_row[k * 2]), float(x_row[k * 2 + 1]), float(x_row[k * 2 + 2]), float(x_row[k * 2 + 3]));
        acc += dot(w, x);
    }
    
    Y[m * dims.y + n] = bfloat(acc);
}
