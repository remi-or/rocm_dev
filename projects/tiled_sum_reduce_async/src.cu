#include <hip/hip_runtime.h>
#include <iostream>

#define WARPSIZE 64
#define WARPTILE_M 8
#define WARPTILE_N 64

#define ELEMS_PER_THREADS ((WARPTILE_M * WARPTILE_N) / WARPSIZE)
#define THREADS_PER_ROW (WARPTILE_N / ELEMS_PER_THREADS)

void __global__ _tiled_sum_reduce_kernel(
    const float* __restrict__ A, 
    const float* __restrict__ B,
    float* __restrict__ D, 
    const int b,
    const int m,
    const int n
) {
    // Relocate current thread
    const int curr_m = (blockIdx.x * WARPTILE_M) + (threadIdx.x % THREADS_PER_ROW);
    const int curr_n = (blockIdx.y * WARPTILE_M) + (threadIdx.x / THREADS_PER_ROW);
    A += curr_m * n + curr_n;
    B += curr_m * n + curr_n;
    D += curr_m * n + curr_n;

    // Initialize results registers
    float reg_D[ELEMS_PER_THREADS];
    #pragma unroll
    for (int i = 0; i < ELEMS_PER_THREADS; i++) {
        reg_D[i] = 0.0f;
    }

    // Main loop
    for (int batch = 0; batch < b; batch++) {
        // Accumulate
        #pragma unroll
        for (int i = 0; i < ELEMS_PER_THREADS; i++) {
            reg_D[i] = A[i] + B[i];
        }
        // Advance
        A += m * n;
        B += m * n;
    }

    // Store
    #pragma unroll
    for (int i = 0; i < ELEMS_PER_THREADS; i++) {
        D[i] = reg_D[i];
    }
}

void tiled_sum_reduce(
    const float* __restrict__ A, 
    const float* __restrict__ B,
    float* __restrict__ D, 
    const int b, 
    const int m, 
    const int n
) {
    // Check shapes
    if ((m % WARPTILE_M != 0) || (n % WARPTILE_N != 0)) {
        std::cerr << "Either m or n is not divisible by the corresponding WARPTILE_" << std::endl;
    }

    // Prepare kernel launch
    const int grid_m = m / WARPTILE_M;
    const int grid_n = n / WARPTILE_N;
    dim3 grid(grid_m, grid_n, 1);
    dim3 block(WARPSIZE, 1, 1);

    // Launch kernel
    _tiled_sum_reduce_kernel<<<grid, block, 0, 0>>>(A, B, D, b, m, n);
}

