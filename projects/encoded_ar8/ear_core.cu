#pragma once

// ------------------------------------------------------ Includes -----------------------------------------------------
#include <vector>

#include <hip/hip_runtime.h>
#include <hip/hip_fp8.h>
#include <hip/hip_fp16.h>
#include <c10/hip/HIPStream.h>

#include <rocprim/rocprim.hpp>

#include <torch/extension.h>

#include <mscclpp/core.hpp>
#include <mscclpp/utils.hpp>
#include <mscclpp/port_channel.hpp>
#include <mscclpp/memory_channel.hpp>
#include <mscclpp/concurrency_device.hpp>



// ----------------------------------------------------- Constants -----------------------------------------------------
#define WARPSIZE 64
#define WORLD_SIZE 8
#define NB_WARPS 16
#define THREADS_PER_BLOCK (NB_WARPS * WARPSIZE)

#define ELEMS_PER_THREAD 16
#define PK_ELEMS_PER_THREAD (ELEMS_PER_THREAD / 2)
#define ELEMS_PER_BLOCK (NB_WARPS * WARPSIZE * ELEMS_PER_THREAD)

#define SATURATE true



// ------------------------------------------------------ Macros -------------------------------------------------------
#define CUDATHROW(cmd)                                                                                                 \
    do {                                                                                                               \
        cudaError_t err = cmd;                                                                                         \
        if (err != cudaSuccess) {                                                                                      \
        std::string msg = std::string("Test CUDA failure: ") + std::string(__FILE__) + ":" + std::to_string(__LINE__)+ \
                            " '" + cudaGetErrorString(err) + "'";                                                      \
        throw std::runtime_error(msg);                                                                                 \
        }                                                                                                              \
    } while (0)



// --------------------------------------------------- Type aliases ----------------------------------------------------
using fp8 = __hip_fp8_storage_t;
using fp8x2 = __hip_fp8x2_storage_t;
using fp8x4 = __hip_fp8x4_storage_t;



// ------------------------------------------- Quantization-related snippets -------------------------------------------
__device__ __forceinline__
float2 dequantizeFp8x2(const fp8x2 &x) {
    union {
        unsigned int i32val;
        fp8x2 i16val[2];
    } val;
    val.i16val[0] = x;

    auto f2 = __builtin_amdgcn_cvt_pk_f32_fp8(val.i32val, false);
    return float2{f2[0], f2[1]};
}

__device__ __forceinline__
float computeNorm(float2 regsFp[PK_ELEMS_PER_THREAD]) {
    // Compute the norm of the loaded elements
    float norm = 0.0f;

    #pragma unroll
    for (int i = 0; i < PK_ELEMS_PER_THREAD; i++) {
        norm += regsFp[i].x * regsFp[i].x + regsFp[i].y * regsFp[i].y;
    }

    // Use warp-reduce to compute the norm of the warp-tile
    using warpReduceFloat = rocprim::warp_reduce<float, WARPSIZE, true>;
    __shared__ warpReduceFloat::storage_type temp[1];
    warpReduceFloat().reduce(norm, norm, temp[0]); // input, output, temp storage

    // Normalize values and store them back
    norm = sqrt(norm / (ELEMS_PER_THREAD * WARPSIZE) + 1e-6f);
    return norm;
}

__device__ __forceinline__
void quantizeAndStore(float2 regsFloat[PK_ELEMS_PER_THREAD], fp8x2* xQuantized, float* xScales)
{
    // Compute norm and inverse scale for faster computation
    float norm = computeNorm(regsFloat);
    float invScale = 1.0f / norm;

    // Quantize to registers
    fp8x2 regsQ[PK_ELEMS_PER_THREAD];
    #pragma unroll
    for (int i = 0; i < PK_ELEMS_PER_THREAD; i++) {
        regsQ[i] = __hip_cvt_float2_to_fp8x2(regsFloat[i] * invScale, __HIP_SATFINITE, __HIP_E4M3_FNUZ);
    }

    // Store quantized values
    #pragma unroll
    for (int i = 0; i < PK_ELEMS_PER_THREAD; i++) {
        xQuantized[i] = regsQ[i];
    }

    // Store the quantization scale
    if (threadIdx.x % WARPSIZE == 0) {
        xScales[0] = norm;
    }
}



// ---------------------------------------------------- Future work ----------------------------------------------------
__device__ __forceinline__
float4 dequantizeAndMulF8x4(const fp8x4* x, const fp8x4* y, float scale) {
    float xTmp[2];
    float yTmp[2];
    float out[4];

    asm volatile(
        "v_cvt_pk_f32_fp8_sdwa %0, %8, src0_sel:WORD_0\n\t"  // xTmp = x[0]
        "v_cvt_pk_f32_fp8_sdwa %1, %9, src0_sel:WORD_0\n\t"  // yTmp = y[0]
        "v_pk_mul_f32 %2, %4, %5\n\t"                        // out[0] = xTmp * yTmp
        "v_cvt_pk_f32_fp8_sdwa %0, %8, src0_sel:WORD_1\n\t"  // xTmp = x[1]
        "v_cvt_pk_f32_fp8_sdwa %1, %9, src0_sel:WORD_1\n\t"  // yTmp = y[1]
        "v_pk_mul_f32 %2, %4, %5\n\t"                        // out[2] = xTmp * yTmp
        "v_mov_b32 %0, scale\n\t"                            // xTmp:low = scale
        "v_pk_mul_f32 %2, %4, %6, op_sel:[0]\n\t"            // out[0] = scale * out[0]
        "v_pk_mul_f32 %3, %4, %7, op_sel:[0]\n\t"            // out[2] = scale * out[2]
        : "=v"(xTmp), "=v"(yTmp), "=v"(out[0]), "=v"(out[2])
        :  "v"(xTmp),  "v"(yTmp),  "v"(out[0]), "v"(out[2]), "v"(x), "v"(y), "v"(scale)
    );
    return reinterpret_cast<float4*>(out)[0];
}
