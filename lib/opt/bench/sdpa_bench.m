/* MPSGraph SDPA 인코드 비용 분리 실험.
 * 질문: 윈도우 어텐션에서 "인코드 195초"의 정체는 무엇인가?
 *  A. 순수 헤드우선 SDPA (밀집 기본 경로와 동일) — 기준선
 *  B. 행우선 + 그래프 내 전치 (플래그 경로의 밀집 — 인코드 569초를 봤던 형태)
 *  C. 슬라이스+연접 윈도우 그래프 (현재 희소 구현)
 *  D. 균일 배치 SDPA, 순수 플레이스홀더 (개선안 v3 의 목표 형태)
 * 각각 인코드 벽시계와 GPU 완료 벽시계를 따로 잰다. */
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>
#include <mach/mach_time.h>

static double now(void) {
    static mach_timebase_info_data_t tb;
    if (!tb.denom) mach_timebase_info(&tb);
    return mach_absolute_time() * (double)tb.numer / tb.denom / 1e9;
}

enum { HEADS = 56, DIM = 128, ROWS = 43664, SINK = 2192,
       CHUNKS = 9, QC = 4608, KC = 16016, REPS = 8 };

static id<MTLBuffer> buf(id<MTLDevice> dev, size_t elems) {
    return [dev newBufferWithLength:elems * 2 options:MTLResourceStorageModeShared];
}
static MPSGraphTensorData *data(id<MTLBuffer> b, NSArray<NSNumber*> *shape) {
    return [[MPSGraphTensorData alloc] initWithMTLBuffer:b shape:shape
                                                dataType:MPSDataTypeBFloat16];
}

static void run(NSString *name, id<MTLDevice> dev, id<MTLCommandQueue> queue,
                MPSGraph *g, NSDictionary *feeds, MPSGraphTensor *out,
                MPSGraphTensorData *outData) {
    double enc = 0, gpu = 0;
    for (int i = 0; i < REPS; i++) {
        @autoreleasepool {
            MPSCommandBuffer *cb = [MPSCommandBuffer commandBufferFromCommandQueue:queue];
            double t0 = now();
            [g encodeToCommandBuffer:cb feeds:feeds targetOperations:nil
                   resultsDictionary:@{out: outData} executionDescriptor:nil];
            double t1 = now();
            [cb commit];
            [cb waitUntilCompleted];
            double t2 = now();
            if (i) { enc += t1 - t0; gpu += t2 - t1; } /* 첫 회는 컴파일 워밍업 */
        }
    }
    printf("  %-28s 인코드 %7.3fs/회   GPU %7.3fs/회\n",
           name.UTF8String, enc / (REPS - 1), gpu / (REPS - 1));
    fflush(stdout);
}

int main(void) { @autoreleasepool {
    id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
    id<MTLCommandQueue> queue = [dev newCommandQueue];
    printf("장치: %s\n", dev.name.UTF8String);
    NSArray *hm = @[@1, @(HEADS), @(ROWS), @(DIM)];
    NSArray *rm = @[@1, @(ROWS), @(HEADS), @(DIM)];
    id<MTLBuffer> q = buf(dev, (size_t)ROWS*HEADS*DIM), k = buf(dev, (size_t)ROWS*HEADS*DIM),
                  v = buf(dev, (size_t)ROWS*HEADS*DIM), o = buf(dev, (size_t)ROWS*HEADS*DIM);
    float scale = 1.0f / sqrtf(DIM);

    { /* A: 순수 헤드우선 밀집 */
        MPSGraph *g = [MPSGraph new];
        MPSGraphTensor *tq = [g placeholderWithShape:hm dataType:MPSDataTypeBFloat16 name:nil];
        MPSGraphTensor *tk = [g placeholderWithShape:hm dataType:MPSDataTypeBFloat16 name:nil];
        MPSGraphTensor *tv = [g placeholderWithShape:hm dataType:MPSDataTypeBFloat16 name:nil];
        MPSGraphTensor *att = [g scaledDotProductAttentionWithQueryTensor:tq keyTensor:tk
                                 valueTensor:tv scale:scale name:nil];
        run(@"A 헤드우선 밀집", dev, queue, g,
            @{tq: data(q,hm), tk: data(k,hm), tv: data(v,hm)}, att, data(o,hm));
    }
    { /* B: 행우선 + 전치 밀집 */
        MPSGraph *g = [MPSGraph new];
        MPSGraphTensor *tq = [g placeholderWithShape:rm dataType:MPSDataTypeBFloat16 name:nil];
        MPSGraphTensor *tk = [g placeholderWithShape:rm dataType:MPSDataTypeBFloat16 name:nil];
        MPSGraphTensor *tv = [g placeholderWithShape:rm dataType:MPSDataTypeBFloat16 name:nil];
        MPSGraphTensor *att = [g scaledDotProductAttentionWithQueryTensor:
              [g transposeTensor:tq dimension:1 withDimension:2 name:nil]
            keyTensor:[g transposeTensor:tk dimension:1 withDimension:2 name:nil]
            valueTensor:[g transposeTensor:tv dimension:1 withDimension:2 name:nil]
            scale:scale name:nil];
        MPSGraphTensor *out = [g transposeTensor:att dimension:1 withDimension:2 name:nil];
        run(@"B 행우선+전치 밀집", dev, queue, g,
            @{tq: data(q,rm), tk: data(k,rm), tv: data(v,rm)}, out, data(o,rm));
    }
    { /* C: 슬라이스+연접 윈도우 (현재 구현 재현: 싱크 + 9청크) */
        MPSGraph *g = [MPSGraph new];
        MPSGraphTensor *tq = [g placeholderWithShape:hm dataType:MPSDataTypeBFloat16 name:nil];
        MPSGraphTensor *tk = [g placeholderWithShape:hm dataType:MPSDataTypeBFloat16 name:nil];
        MPSGraphTensor *tv = [g placeholderWithShape:hm dataType:MPSDataTypeBFloat16 name:nil];
        MPSGraphTensor *(^sl)(MPSGraphTensor*, int, int) = ^(MPSGraphTensor *t, int a, int len) {
            return [g sliceTensor:t dimension:2 start:a length:len name:nil]; };
        NSMutableArray *parts = [NSMutableArray array];
        [parts addObject:[g scaledDotProductAttentionWithQueryTensor:sl(tq,0,SINK)
                            keyTensor:tk valueTensor:tv scale:scale name:nil]];
        for (int c = 0; c < CHUNKS; c++) {
            int qa = SINK + c*QC, wa = MAX(SINK, qa - 4608), wl = MIN(ROWS - wa, KC - SINK);
            MPSGraphTensor *kk = [g concatTensors:@[sl(tk,0,SINK), sl(tk,wa,wl)] dimension:2 name:nil];
            MPSGraphTensor *vv = [g concatTensors:@[sl(tv,0,SINK), sl(tv,wa,wl)] dimension:2 name:nil];
            [parts addObject:[g scaledDotProductAttentionWithQueryTensor:
                sl(tq,qa,MIN(QC,ROWS-qa)) keyTensor:kk valueTensor:vv scale:scale name:nil]];
        }
        MPSGraphTensor *out = [g concatTensors:parts dimension:2 name:nil];
        run(@"C 슬라이스+연접 윈도우", dev, queue, g,
            @{tq: data(q,hm), tk: data(k,hm), tv: data(v,hm)}, out, data(o,hm));
    }
    { /* D: 균일 배치, 순수 플레이스홀더 (v3 목표) */
        NSArray *bq = @[@(CHUNKS), @(HEADS), @(QC), @(DIM)];
        NSArray *bk = @[@(CHUNKS), @(HEADS), @(KC), @(DIM)];
        id<MTLBuffer> gq = buf(dev, (size_t)CHUNKS*HEADS*QC*DIM);
        id<MTLBuffer> gk = buf(dev, (size_t)CHUNKS*HEADS*KC*DIM);
        id<MTLBuffer> gv = buf(dev, (size_t)CHUNKS*HEADS*KC*DIM);
        id<MTLBuffer> go = buf(dev, (size_t)CHUNKS*HEADS*QC*DIM);
        MPSGraph *g = [MPSGraph new];
        MPSGraphTensor *tq = [g placeholderWithShape:bq dataType:MPSDataTypeBFloat16 name:nil];
        MPSGraphTensor *tk = [g placeholderWithShape:bk dataType:MPSDataTypeBFloat16 name:nil];
        MPSGraphTensor *tv = [g placeholderWithShape:bk dataType:MPSDataTypeBFloat16 name:nil];
        MPSGraphTensor *att = [g scaledDotProductAttentionWithQueryTensor:tq keyTensor:tk
                                 valueTensor:tv scale:scale name:nil];
        run(@"D 균일 배치 (v3 목표)", dev, queue, g,
            @{tq: data(gq,bq), tk: data(gk,bk), tv: data(gv,bk)}, att, data(go,bq));
    }
    return 0;
} }
