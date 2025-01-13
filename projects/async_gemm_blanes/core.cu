#pragma once

#include <hip/hip_runtime.h>
#include <iostream>

// #include <ATen/cuda/CUDAContext.h>
// #include <c10/cuda/CUDAGuard.h>
// #include <hip/hip_fp8.h>
// #include <torch/all.h>
// using fp8 = __hip_fp8_storage_t;
// using fp8x8 = __attribute__( (__vector_size__(8 * sizeof(fp8)) )) fp8;
// using f32x4 = __attribute__( (__vector_size__(4 * sizeof(float)) )) float;
// using uint8 = unsigned char;
// using uint16 = unsigned short;

#define WARPSIZE 64
#define OP_PER_WARPTILE 2
#define OP_MN 16
#define OP_K 32

#define B_LANES 3

#define WARPTILE_M OP_MN
#define WARPTILE_N (OP_MN * B_LANES)
#define WARPTILE_K (OP_K * OP_PER_WARPTILE)
#define PRODUCED_MASK 257

#define PRODUCERS 3
#define CONSUMERS 1
#define QSIZE 12
#define G_ATOMICS true
#define SPLIT_K 2

#define ELEMS_PER_THREADS ((WARPTILE_M * WARPTILE_N) / WARPSIZE)
#define THREADS_PER_ROW (WARPTILE_N / ELEMS_PER_THREADS)


#define K_BLOCKS(k) (((k / WARPTILE_K) / SPLIT_K))

int inline __device__ infer_k_blocks(const int &k) {
    if (SPLIT_K == 1) {
        return k / WARPTILE_K;
    } else {
        if (blockIdx.z < SPLIT_K - 1) {
            return K_BLOCKS(k);
        } else {
            return (k / WARPTILE_K) - (SPLIT_K - 1) * K_BLOCKS(k);
        }
    }
}
