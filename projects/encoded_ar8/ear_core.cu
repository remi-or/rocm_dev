
#define WARPSIZE 64
#define WORLD_SIZE 8
#define NB_WARPS 16

#define ELEMS_PER_THREAD 16
#define PK_ELEMS_PER_THREAD (ELEMS_PER_THREAD / 2)
#define ELEMS_PER_BLOCK (NB_WARPS * WARPSIZE * ELEMS_PER_THREAD)

#define SATURATE true

__device__ __forceinline__ float computeNorm(float2 regsFp[PK_ELEMS_PER_THREAD]) {
    // Compute the norm of the loaded elements
    float norm = 0.0f

    #pragma unroll
    for (int i = 0; i < PK_ELEMS_PER_THREAD; i++) {
        norm += regsFp[i].x * regsFp[i].x + regsFp[i].y * regsFp[i].y;
    }

    // Use warp-reduce to compute the norm of the warp-tile
    using warpReduceFloat = rocprim::warp_reduce<float, WARPSIZE, true>;
    __shared__ warpReduceFloat::storage_type temp[1];
    warpReduceFloat().reduce(norm, norm, temp[0]); // input, output, temp storage

    // Normalize values and store them back
    norm = sqrt(norm / (ELEMS_PER_THREAD * WARPSIZE));
    return norm;
}

template<typename T>
__device__ __forceinline__ void load128b(T* regs, T* src) {
    #pragma unroll
    for (int i = 0; i < 16 / sizeof(T); i++) {
        regs[i] = src[i];
    }
}

__device__ __forceinline__ void quantizeAndStore(float2 regsFp[PK_ELEMS_PER_THREAD], fp8x2* xQuantized, float* xScales)
{
    // Compute norm and inverse scale for faster computation
    float norm = computeNorm(regsFp);
    float invScale = 1.0f / norm;

    // Quantize to registers
    fp8x2 regsQ[PK_ELEMS_PER_THREAD];
#pragma unroll
    for (int i = 0; i < PK_ELEMS_PER_THREAD; i++) {
        regsQ[i] = cast_to_f8x2_from_f32x2(regsFp[i] * invScale, SATURATE, __HIP_E4M3_FNUZ);
    }

    // Store quantized values
#pragma unroll
    for (int i = 0; i < PK_ELEMS_PER_THREAD; i++) {
        xQuantized[i] = regsQ[i];
    }

    // Store the quantization scale
    if (threadId % WARPSIZE == 0) {
        xScales[0] = norm;
    }
}
