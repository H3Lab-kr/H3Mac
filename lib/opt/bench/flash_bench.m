/* 윈도우 FlashAttention 정확성·속도 검증 벤치.
 * 기준: MPSGraph SDPA 로 [싱크|창] 을 연접해 계산한 결과.
 * 비교: 직접 작성한 커널 (복사 없이 두 구간을 순회). */
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>
#include <mach/mach_time.h>

static double now(void){ static mach_timebase_info_data_t t; if(!t.denom) mach_timebase_info(&t);
    return mach_absolute_time()*(double)t.numer/t.denom/1e9; }
static uint16_t f2b(float f){ uint32_t u; memcpy(&u,&f,4); return (uint16_t)(u>>16); }
static float b2f(uint16_t h){ uint32_t u=(uint32_t)h<<16; float f; memcpy(&f,&u,4); return f; }

int main(int argc,char**argv){ @autoreleasepool{
    uint32_t HEADS = argc>1?atoi(argv[1]):8;
    uint32_t ROWS  = argc>2?atoi(argv[2]):4096;
    uint32_t SINK  = argc>3?atoi(argv[3]):512;
    uint32_t W0    = argc>4?atoi(argv[4]):1024;
    uint32_t W1    = argc>5?atoi(argv[5]):2048;
    uint32_t Q0    = argc>6?atoi(argv[6]):1024;
    uint32_t QC    = argc>7?atoi(argv[7]):512;
    const uint32_t D = 128;
    float scale = 1.0f/sqrtf((float)D);
    printf("헤드 %u · 행 %u · 싱크 %u · 창 [%u,%u) · 쿼리 [%u,+%u)\n",
           HEADS,ROWS,SINK,W0,W1,Q0,QC);

    id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
    id<MTLCommandQueue> q = [dev newCommandQueue];
    size_t n = (size_t)HEADS*ROWS*D;
    id<MTLBuffer> bq=[dev newBufferWithLength:n*2 options:0],
                  bk=[dev newBufferWithLength:n*2 options:0],
                  bv=[dev newBufferWithLength:n*2 options:0],
                  bo=[dev newBufferWithLength:n*2 options:0];
    uint16_t *pq=bq.contents,*pk=bk.contents,*pv=bv.contents;
    srandom(42);
    for(size_t i=0;i<n;i++){
        pq[i]=f2b(((float)random()/RAND_MAX-0.5f)*2.f);
        pk[i]=f2b(((float)random()/RAND_MAX-0.5f)*2.f);
        pv[i]=f2b(((float)random()/RAND_MAX-0.5f)*2.f);
    }
    /* ── 기준: CPU 로 정확 계산 (쿼리 몇 행만) ── */
    uint32_t CHECK=4;
    float *ref = calloc((size_t)CHECK*HEADS*D,sizeof(float));
    for(uint32_t h=0;h<HEADS;h++) for(uint32_t c=0;c<CHECK;c++){
        uint32_t qr=Q0+c*(QC/CHECK);
        const uint16_t *Qp=pq+((size_t)h*ROWS+qr)*D;
        double mx=-1e30,sum=0;
        uint32_t total=SINK+(W1-W0);
        double *sc=malloc(total*sizeof(double)); uint32_t idx=0;
        for(uint32_t seg=0;seg<2;seg++){
            uint32_t s0=seg?W0:0,s1=seg?W1:SINK;
            for(uint32_t k=s0;k<s1;k++){
                const uint16_t *Kp=pk+((size_t)h*ROWS+k)*D; double d=0;
                for(uint32_t j=0;j<D;j++) d+=b2f(Qp[j])*(double)b2f(Kp[j]);
                d*=scale; sc[idx++]=d; if(d>mx)mx=d;
            }
        }
        for(uint32_t i=0;i<idx;i++){ sc[i]=exp(sc[i]-mx); sum+=sc[i]; }
        idx=0;
        for(uint32_t seg=0;seg<2;seg++){
            uint32_t s0=seg?W0:0,s1=seg?W1:SINK;
            for(uint32_t k=s0;k<s1;k++){
                double w=sc[idx++]/sum;
                const uint16_t *Vp=pv+((size_t)h*ROWS+k)*D;
                for(uint32_t j=0;j<D;j++) ref[((size_t)c*HEADS+h)*D+j]+=w*b2f(Vp[j]);
            }
        }
        free(sc);
    }
    /* ── 커널 컴파일 ── */
    NSError *err=nil;
    NSString *src=[NSString stringWithContentsOfFile:@"flash_kernel.metal"
                    encoding:NSUTF8StringEncoding error:&err];
    if(!src){ printf("커널 소스 없음\n"); return 1; }
    MTLCompileOptions *co=[MTLCompileOptions new];
    const char *abl=getenv("ABLATE");
    co.preprocessorMacros=@{@"ABLATE": abl?@(atoi(abl)):@0};
    id<MTLLibrary> lib=[dev newLibraryWithSource:src options:co error:&err];
    if(!lib){ printf("컴파일 실패: %s\n", err.localizedDescription.UTF8String); return 1; }
    id<MTLComputePipelineState> ps=[dev newComputePipelineStateWithFunction:
        [lib newFunctionWithName:@"h3_flash_window_bf16"] error:&err];
    if(!ps){ printf("파이프라인 실패: %s\n", err.localizedDescription.UTF8String); return 1; }
    printf("장치 threadgroup 최대 %lu바이트\n", (unsigned long)dev.maxThreadgroupMemoryLength);
    printf("커널 컴파일 OK (최대 스레드 %lu · threadgroup 메모리 %lu)\n",
           (unsigned long)ps.maxTotalThreadsPerThreadgroup,
           (unsigned long)ps.staticThreadgroupMemoryLength);

    struct { uint32_t heads,rows,sink,w0,w1,q0,q_count; float scale; }
        args={HEADS,ROWS,SINK,W0,W1,Q0,QC,scale};
    size_t shmem = 31232;
    double best=1e9;
    for(int rep=0;rep<5;rep++){
        id<MTLCommandBuffer> cb=[q commandBuffer];
        id<MTLComputeCommandEncoder> e=[cb computeCommandEncoder];
        [e setComputePipelineState:ps];
        [e setBuffer:bq offset:0 atIndex:0]; [e setBuffer:bk offset:0 atIndex:1];
        [e setBuffer:bv offset:0 atIndex:2]; [e setBuffer:bo offset:0 atIndex:3];
        [e setBytes:&args length:sizeof(args) atIndex:4];
        [e setThreadgroupMemoryLength:shmem atIndex:0];
        [e dispatchThreadgroups:MTLSizeMake((QC+63)/64,HEADS,1)
            threadsPerThreadgroup:MTLSizeMake(256,1,1)];
        [e endEncoding];
        double t0=now(); [cb commit]; [cb waitUntilCompleted];
        double dt=now()-t0; if(rep&&dt<best)best=dt;
        if(cb.error){ printf("실행 오류: %s\n", cb.error.localizedDescription.UTF8String); return 1; }
    }
    uint16_t *po=bo.contents;
    double maxerr=0,sumref=0;
    for(uint32_t h=0;h<HEADS;h++) for(uint32_t c=0;c<CHECK;c++){
        uint32_t qr=Q0+c*(QC/CHECK);
        for(uint32_t j=0;j<D;j++){
            double got=b2f(po[((size_t)h*ROWS+qr)*D+j]);
            double want=ref[((size_t)c*HEADS+h)*D+j];
            maxerr=fmax(maxerr,fabs(got-want)); sumref+=fabs(want);
        }
    }
    printf("\n최대 절대 오차 %.5f   (기준 평균크기 %.5f)\n",
           maxerr, sumref/((double)HEADS*CHECK*D));
    printf("커널 시간 %.4f ms\n", best*1e3);
    printf("%s\n", maxerr < 0.02 ? "✓ 정확성 통과" : "✗ 정확성 실패");

    /* ── 동일 형상 MPSGraph 대조 (싱크|창 을 연접해 계산) ── */
    {
        uint32_t KV = SINK + (W1-W0);
        id<MTLBuffer> gk=[dev newBufferWithLength:(size_t)HEADS*KV*D*2 options:0];
        id<MTLBuffer> gv=[dev newBufferWithLength:(size_t)HEADS*KV*D*2 options:0];
        id<MTLBuffer> gq=[dev newBufferWithLength:(size_t)HEADS*QC*D*2 options:0];
        id<MTLBuffer> go=[dev newBufferWithLength:(size_t)HEADS*QC*D*2 options:0];
        MPSGraph *g=[MPSGraph new];
        NSArray *qs=@[@1,@(HEADS),@(QC),@(D)], *ks=@[@1,@(HEADS),@(KV),@(D)];
        MPSGraphTensor *tq=[g placeholderWithShape:qs dataType:MPSDataTypeBFloat16 name:nil];
        MPSGraphTensor *tk=[g placeholderWithShape:ks dataType:MPSDataTypeBFloat16 name:nil];
        MPSGraphTensor *tv=[g placeholderWithShape:ks dataType:MPSDataTypeBFloat16 name:nil];
        MPSGraphTensor *att=[g scaledDotProductAttentionWithQueryTensor:tq keyTensor:tk
                               valueTensor:tv scale:scale name:nil];
        MPSGraphTensorData *dq=[[MPSGraphTensorData alloc] initWithMTLBuffer:gq shape:qs dataType:MPSDataTypeBFloat16];
        MPSGraphTensorData *dk=[[MPSGraphTensorData alloc] initWithMTLBuffer:gk shape:ks dataType:MPSDataTypeBFloat16];
        MPSGraphTensorData *dv=[[MPSGraphTensorData alloc] initWithMTLBuffer:gv shape:ks dataType:MPSDataTypeBFloat16];
        MPSGraphTensorData *dov=[[MPSGraphTensorData alloc] initWithMTLBuffer:go shape:qs dataType:MPSDataTypeBFloat16];
        double mbest=1e9;
        for(int rep=0;rep<5;rep++){
            @autoreleasepool{
                MPSCommandBuffer *cb=[MPSCommandBuffer commandBufferFromCommandQueue:q];
                [g encodeToCommandBuffer:cb feeds:@{tq:dq,tk:dk,tv:dv} targetOperations:nil
                       resultsDictionary:@{att:dov} executionDescriptor:nil];
                double t0=now(); [cb commit]; [cb waitUntilCompleted];
                double dt=now()-t0; if(rep&&dt<mbest)mbest=dt;
            }
        }
        printf("MPSGraph 동일형상 %.4f ms   (커널 대비 %.2f배)\n", mbest*1e3, best/mbest);
    }
    return 0;

} }
