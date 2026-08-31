#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <mach/mach_time.h>
#import <math.h>
#import <stdlib.h>

static uint16_t f2b(float f) {
    uint32_t x; memcpy(&x, &f, 4);
    return (uint16_t)(x >> 16);
}
static float b2f(uint16_t b) {
    uint32_t x = ((uint32_t)b) << 16;
    float f; memcpy(&f, &x, 4);
    return f;
}
static double now_sec(void) {
    static mach_timebase_info_data_t tb;
    if (!tb.denom) mach_timebase_info(&tb);
    return (double)mach_absolute_time() * tb.numer / tb.denom * 1e-9;
}

int main(void) {
    @autoreleasepool {
        id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
        if (!dev) { printf("Metal 미지원 기기\n"); return 1; }
        printf("장치: %s (통합 메모리 %.1f GB)\n",
               dev.name.UTF8String, (double)dev.recommendedMaxWorkingSetSize / (1024*1024*1024));

        NSError *err = nil;
        NSString *src = [NSString stringWithContentsOfFile:@"quant_bench.metal"
                                                 encoding:NSUTF8StringEncoding error:&err];
        if (!src) { printf("소스 읽기 실패: %s\n", err.localizedDescription.UTF8String); return 1; }
        
        id<MTLLibrary> lib = [dev newLibraryWithSource:src options:nil error:&err];
        if (!lib) { printf("컴파일 실패: %s\n", err.localizedDescription.UTF8String); return 1; }

        id<MTLComputePipelineState> ps_i8 = [dev newComputePipelineStateWithFunction:
            [lib newFunctionWithName:@"h3_int8_linear_bench"] error:&err];
        id<MTLComputePipelineState> ps_i4 = [dev newComputePipelineStateWithFunction:
            [lib newFunctionWithName:@"h3_int4_linear_bench"] error:&err];
        if (!ps_i8 || !ps_i4) { printf("파이프라인 생성 실패\n"); return 1; }

        // DiT Linear 프로젝션 실측 크기 (M: 토큰 수 512, N: 출력 3072, K: 입력 3072)
        const uint32_t M = 512, N = 3072, K = 3072;
        struct { uint32_t m, n, k; } dims = {M, N, K};

        size_t bytes_w_i8 = (size_t)N * K;                  // 9.4 MB
        size_t bytes_w_i4 = (size_t)N * (K / 2);            // 4.7 MB (50% 절감!)
        size_t bytes_s_i8 = (size_t)N * sizeof(float);
        size_t bytes_s_i4 = (size_t)N * (K / 128) * sizeof(float);
        size_t bytes_x = (size_t)M * K * sizeof(uint16_t);
        size_t bytes_y = (size_t)M * N * sizeof(uint16_t);

        id<MTLBuffer> buf_w_i8 = [dev newBufferWithLength:bytes_w_i8 options:MTLResourceStorageModeShared];
        id<MTLBuffer> buf_w_i4 = [dev newBufferWithLength:bytes_w_i4 options:MTLResourceStorageModeShared];
        id<MTLBuffer> buf_s_i8 = [dev newBufferWithLength:bytes_s_i8 options:MTLResourceStorageModeShared];
        id<MTLBuffer> buf_s_i4 = [dev newBufferWithLength:bytes_s_i4 options:MTLResourceStorageModeShared];
        id<MTLBuffer> buf_x = [dev newBufferWithLength:bytes_x options:MTLResourceStorageModeShared];
        id<MTLBuffer> buf_y_i8 = [dev newBufferWithLength:bytes_y options:MTLResourceStorageModeShared];
        id<MTLBuffer> buf_y_i4 = [dev newBufferWithLength:bytes_y options:MTLResourceStorageModeShared];
        id<MTLBuffer> buf_dims = [dev newBufferWithBytes:&dims length:sizeof(dims) options:MTLResourceStorageModeShared];

        // 랜덤 데이터 초기화
        int8_t *pw_i8 = buf_w_i8.contents;
        uint8_t *pw_i4 = buf_w_i4.contents;
        float *pscales_i8 = buf_s_i8.contents;
        float *pscales_i4 = buf_s_i4.contents;
        uint16_t *px = buf_x.contents;

        for (size_t i = 0; i < bytes_w_i8; i++) pw_i8[i] = (int8_t)((rand() % 255) - 127);
        for (size_t i = 0; i < bytes_w_i4; i++) pw_i4[i] = (uint8_t)(rand() % 256);
        for (size_t i = 0; i < N; i++) pscales_i8[i] = 0.005f + (float)rand() / RAND_MAX * 0.01f;
        for (size_t i = 0; i < N * (K / 128); i++) pscales_i4[i] = 0.005f + (float)rand() / RAND_MAX * 0.01f;
        for (size_t i = 0; i < M * K; i++) px[i] = f2b(((float)rand() / (float)RAND_MAX - 0.5f) * 2.0f);

        id<MTLCommandQueue> queue = [dev newCommandQueue];

        // 1. INT8 벤치마크 (5회 반복 최저시간)
        double best_i8 = 1e9;
        for (int rep = 0; rep < 10; rep++) {
            id<MTLCommandBuffer> cb = [queue commandBuffer];
            id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
            [enc setComputePipelineState:ps_i8];
            [enc setBuffer:buf_w_i8 offset:0 atIndex:0];
            [enc setBuffer:buf_s_i8 offset:0 atIndex:1];
            [enc setBuffer:buf_x offset:0 atIndex:2];
            [enc setBuffer:buf_y_i8 offset:0 atIndex:3];
            [enc setBuffer:buf_dims offset:0 atIndex:4];
            [enc dispatchThreads:MTLSizeMake(M, N, 1) threadsPerThreadgroup:MTLSizeMake(16, 16, 1)];
            [enc endEncoding];
            double t0 = now_sec();
            [cb commit];
            [cb waitUntilCompleted];
            double dt = (now_sec() - t0) * 1000.0;
            if (rep > 1 && dt < best_i8) best_i8 = dt;
        }

        // 2. INT4 벤치마크 (5회 반복 최저시간)
        double best_i4 = 1e9;
        for (int rep = 0; rep < 10; rep++) {
            id<MTLCommandBuffer> cb = [queue commandBuffer];
            id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
            [enc setComputePipelineState:ps_i4];
            [enc setBuffer:buf_w_i4 offset:0 atIndex:0];
            [enc setBuffer:buf_s_i4 offset:0 atIndex:1];
            [enc setBuffer:buf_x offset:0 atIndex:2];
            [enc setBuffer:buf_y_i4 offset:0 atIndex:3];
            [enc setBuffer:buf_dims offset:0 atIndex:4];
            [enc dispatchThreads:MTLSizeMake(M, N, 1) threadsPerThreadgroup:MTLSizeMake(16, 16, 1)];
            [enc endEncoding];
            double t0 = now_sec();
            [cb commit];
            [cb waitUntilCompleted];
            double dt = (now_sec() - t0) * 1000.0;
            if (rep > 1 && dt < best_i4) best_i4 = dt;
        }

        // 3. 모델 전체 크기 환산 (33B 가중치 기준)
        double model_size_fp16 = 66.0;
        double model_size_int8 = 33.0;
        double model_size_int4 = 16.5;

        printf("\n==========================================================\n");
        printf(" 📊 H3Lab INT8 vs INT4 실측 비교 분석 결과\n");
        printf("==========================================================\n");
        printf(" 1) 모델 가중치 용량:\n");
        printf("    - INT8 : %.1f GB (현재)\n", model_size_int8);
        printf("    - INT4 : %.1f GB (50%% 추가 압축 ➔ 32GB Mac 가능)\n", model_size_int4);
        printf("\n 2) 3072x3072 DiT 프로젝션 연산 속도 (단일 레이어):\n");
        printf("    - INT8 GEMV : %.3f ms\n", best_i8);
        printf("    - INT4 GEMV : %.3f ms (언패킹 ALU 연산 포함)\n", best_i4);
        printf("    - 속도 배율 : %.2f 배 (%s)\n",
               best_i8 / best_i4,
               (best_i4 <= best_i8) ? "INT4 소폭 빠름" : "INT8이 더 빠름 (비트 언패킹 오버헤드)");
        printf("==========================================================\n");
    }
    return 0;
}
