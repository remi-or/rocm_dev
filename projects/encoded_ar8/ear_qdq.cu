#include "./ear_core.cu"

__device__ void warpwiseQuantize(
    const half2* __restrict__ xHalf,
    fp8x2* __restrict__ xQuantized,
    float* __restrict__ xScales
) {
    const int threadId = threadIdx.x;
    const int warpId = threadIdx.x % WARPSIZE;

    // Relocate in sources
    xHalf += threadId * PK_ELEMS_PER_THREAD;
    xQuantized += threadId * PK_ELEMS_PER_THREAD;
    xScales += warpId;

    // Each thread loads ELEMS_PER_THREAD elements (so half of that in packed elements)
    half2 regsHp[PK_ELEMS_PER_THREAD];
    #pragma unroll
    for (int i = 0; i < PK_ELEMS_PER_THREAD; i++) { regsHp[i] = xHalf[i]; }

    // Convert to fp32
    float2 regsFp[PK_ELEMS_PER_THREAD];
    #pragma unroll
    for (int i = 0; i < PK_ELEMS_PER_THREAD; i++) { regsFp[i] = __half22float2(regsHp[i]); }

    // Quantize and store
    quantizeAndStore(regsFp, xQuantized, xScales);
}


__device__ void warpwiseDequantize(
    fp8x2* xQuantized,
    float* xScales,
    half2* output
) {
    const int threadId = threadIdx.x;
    const int warpId = threadIdx.x % WARPSIZE;

    // Relocate in sources
    xQuantized += threadId * PK_ELEMS_PER_THREAD;

    // Each thread loads ELEMS_PER_THREAD elements (so half of that in packed elements)
    fp8x2 xRegs[PK_ELEMS_PER_THREAD];
    #pragma unroll
    for (int i = 0; i < PK_ELEMS_PER_THREAD; i++) { xRegs[i] = xQuantized[i]; }

    // Also load the scales for x and y
    const float scale = xScales[warpId];

    // Dequantize to fp32, scale and convert to fp16
    half2 regsHp[PK_ELEMS_PER_THREAD];

    #pragma unroll
    for (int i = 0; i < PK_ELEMS_PER_THREAD; i++) {
        float2 tmp = cast_to_f32x2_from_f8x2(reinterpret_cast<fp8_2>(xRegs[i]), __HIP_E4M3_FNUZ);
        tmp *= scale;
        regsHp[i] = __float22half2(tmp);
    }

    // Store back
    #pragma unroll
    for (int i = 0; i < PK_ELEMS_PER_THREAD; i++) {
        output[i] = regsHp[i];
    }
}


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
        float2 xy = cast_to_f32x2_from_f8x2(reinterpret_cast<fp8_2>(xRegs[i]), __HIP_E4M3_FNUZ);
        xy *= cast_to_f32x2_from_f8x2(reinterpret_cast<fp8_2>(yRegs[i]), __HIP_E4M3_FNUZ);
        xy *= scale;
        xyRegs[i] = __float22half2(xy);
    }

    // Store back
    #pragma unroll
    for (int i = 0; i < PK_ELEMS_PER_THREAD; i++) {
        output[i] = xyRegs[i];
    }
}


__device__ void warpwiseInplaceDQAQ( // Dequantize, Accumulate, Quantize in place
    fp8x2* xQuantized,
    const fp8x2* yQuantized,
    float* xScales,
    const float* yScales,
) {
    const int threadId = threadIdx.x;
    const int warpId = threadIdx.x % WARPSIZE;

    // Relocate thread in sources
    xQuantized += threadId * PK_ELEMS_PER_THREAD;
    yQuantized += threadId * PK_ELEMS_PER_THREAD;
    xScales += warpId;
    yScales += warpId;

    // Each thread loads ELEMS_PER_THREAD elements (so half of that in packed elements)
    fp8x2 xRegs[PK_ELEMS_PER_THREAD];
    #pragma unroll
    for (int i = 0; i < PK_ELEMS_PER_THREAD; i++) { xRegs[i] = xQuantized[i]; }

    fp8x2 yRegs[PK_ELEMS_PER_THREAD];
    #pragma unroll
    for (int i = 0; i < PK_ELEMS_PER_THREAD; i++) { yRegs[i] = yQuantized[i]; }

    // Also load the scales for x and y
    const float scale = xScales[0] * yScales[0];

    // Dequantize to fp32
    float2 regsFp[PK_ELEMS_PER_THREAD];
#pragma unroll
    for (int i = 0; i < PK_ELEMS_PER_THREAD; i++) {
        regsFp[i] = cast_to_f32x2_from_f8x2(reinterpret_cast<fp8_2>(xRegs[i]), __HIP_E4M3_FNUZ);
        regsFp[i] *= cast_to_f32x2_from_f8x2(reinterpret_cast<fp8_2>(yRegs[i]), __HIP_E4M3_FNUZ);
        regsFp[i] *= scale;
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
