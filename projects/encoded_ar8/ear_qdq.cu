#include "./ear_core.cu"

__device__ void warpwiseQuantize(
    const half2* __restrict__ xHalf,
    fp8x2* __restrict__ xQuantized,
    float* __restrict__ xScales
) {
    // Each thread loads ELEMS_PER_THREAD elements (so half of that in packed elements)
    half2 regsHalf[PK_ELEMS_PER_THREAD];
    #pragma unroll
    for (int i = 0; i < PK_ELEMS_PER_THREAD; i++) { regsHalf[i] = xHalf[i]; }

    // Convert to fp32
    float2 regsFloat[PK_ELEMS_PER_THREAD];
    #pragma unroll
    for (int i = 0; i < PK_ELEMS_PER_THREAD; i++) { regsFloat[i] = __half22float2(regsHalf[i]); }

    // Quantize and store
    quantizeAndStore(regsFloat, xQuantized, xScales);
}


__device__ void warpwiseDequantize(
    const fp8x2* xQuantized,
    const float* xScales,
    half2* output
) {
    // Each thread loads ELEMS_PER_THREAD elements (so half of that in packed elements)
    fp8x2 xRegs[PK_ELEMS_PER_THREAD];
    #pragma unroll
    for (int i = 0; i < PK_ELEMS_PER_THREAD; i++) { xRegs[i] = xQuantized[i]; }

    // Also load the scales for x and y
    const float scale = xScales[0];

    // Dequantize to fp32, scale and convert to fp16
    half2 regsHalf[PK_ELEMS_PER_THREAD];

    #pragma unroll
    for (int i = 0; i < PK_ELEMS_PER_THREAD; i++) {
        float2 tmp = dequantizeFp8x2(xRegs[i]);
        tmp *= scale;
        tmp = float2{tmp.x, tmp.y};
        regsHalf[i] = __float22half2_rn(tmp);
    }

    // Store back
    #pragma unroll
    for (int i = 0; i < PK_ELEMS_PER_THREAD; i++) {
        output[i] = regsHalf[i];
    }
}

__device__ void warpwiseInplaceDQAQ( // Dequantize, Accumulate, Quantize in place
    fp8x2* xQuantized,
    const fp8x2* yQuantized,
    float* xScales,
    const float* yScales
) {
    // Each thread loads ELEMS_PER_THREAD elements (so half of that in packed elements)
    fp8x2 xRegs[PK_ELEMS_PER_THREAD];
    #pragma unroll
    for (int i = 0; i < PK_ELEMS_PER_THREAD; i++) { xRegs[i] = xQuantized[i]; }

    fp8x2 yRegs[PK_ELEMS_PER_THREAD];
    #pragma unroll
    for (int i = 0; i < PK_ELEMS_PER_THREAD; i++) { yRegs[i] = yQuantized[i]; }

    // Also load the scales for x and y
    const float xScale = xScales[0];
    const float yScale = yScales[0];

    // Dequantize to fp32
    float2 regsFp[PK_ELEMS_PER_THREAD];
#pragma unroll
    for (int i = 0; i < PK_ELEMS_PER_THREAD; i++) {
        regsFp[i] = dequantizeFp8x2(xRegs[i]) * xScale + dequantizeFp8x2(yRegs[i]) * yScale;
    }

    // Store back as quantized values
    quantizeAndStore(regsFp, xQuantized, xScales);
}

// 0 1 2 3
// 01 01 23 23
// 0123 0123 0123 0123

// 0 1 2 3
// 01 12 23 30
// 012 123 230 301
// 0123 1230 2301 3012

// 0 1 2 3 4 5 6 7
// 01 01 23 23 45 45 67 67
// 0123 0123 2345 2345 4567 4567 6701 6701
// 01234567 01234567 01234567 01234567 01234567 01234567 01234567 01234567


__device__ void warpwiseDequantizeAccumulate(
    fp8x2* xQuantized,
    fp8x2* yQuantized,
    float* xScales,
    float* yScales,
    half2* output
) {
    const int threadId = threadIdx.x;
    const int warpId = threadIdx.x % WARPSIZE;

    // Relocate in sources
    xQuantized += threadId * PK_ELEMS_PER_THREAD;
    yQuantized += threadId * PK_ELEMS_PER_THREAD;

    // Each thread loads ELEMS_PER_THREAD elements (so half of that in packed elements)
    fp8x2 xRegs[PK_ELEMS_PER_THREAD];
    #pragma unroll
    for (int i = 0; i < PK_ELEMS_PER_THREAD; i++) { xRegs[i] = xQuantized[i]; }

    fp8x2 yRegs[PK_ELEMS_PER_THREAD];
    #pragma unroll
    for (int i = 0; i < PK_ELEMS_PER_THREAD; i++) { yRegs[i] = yQuantized[i]; }

    // Also load the scales for x and y
    const float scale = xScales[warpId] * yScales[warpId];

    // Dequantize to fp32, scale and convert to fp16
    half2 xyRegs[PK_ELEMS_PER_THREAD];

    #pragma unroll
    for (int i = 0; i < PK_ELEMS_PER_THREAD; i++) {
        // TODO: try dq and mul
        float2 xy = dequantizeFp8x2(xRegs[i]) * dequantizeFp8x2(yRegs[i]);
        xy *= scale;
        xyRegs[i] = __float22half2_rn(xy);
    }

    // Store back
    #pragma unroll
    for (int i = 0; i < PK_ELEMS_PER_THREAD; i++) {
        output[i] = xyRegs[i];
    }
}
