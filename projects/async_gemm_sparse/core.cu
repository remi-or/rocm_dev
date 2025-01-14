#pragma once

#include <hip/hip_runtime.h>
#include <iostream>

// #include <ATen/cuda/CUDAContext.h>
// #include <c10/cuda/CUDAGuard.h>
// #include <hip/hip_fp8.h>
// #include <torch/all.h>
// using fp8 = __hip_fp8_storage_t;
// using fp8x8 = __attribute__( (__vector_size__(8 * sizeof(fp8)) )) fp8;
// using fp8x16 = __attribute__( (__vector_size__(16 * sizeof(fp8)) )) fp8;
// using fp8_4x2 = __attribute__( (__vector_size__(2 * sizeof(int)) )) int;
// using fp8_4x4 = __attribute__( (__vector_size__(4 * sizeof(int)) )) int;
// using f32x4 = __attribute__( (__vector_size__(4 * sizeof(float)) )) float;
// using uint8 = unsigned char;
// using uint16 = unsigned short;
// using uint32 = unsigned int;
// using uint64 = unsigned long long;

#define WARPSIZE 64
#define OP_M 8
#define OP_N 16
#define OP_K 64

#define SMEM_BANKS 32
#define E_PER_BANK 4

#define A_PRODUCERS 2
#define B_PRODUCERS 3
#define CONSUMERS 2

#define WARPTILE_M OP_M
#define WARPTILE_N (OP_N * 4)
#define WARPTILE_K OP_K
#define PRODUCED_MASK 257

#define QSIZE 24
#define G_ATOMICS true
#define SPLIT_K 1

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
