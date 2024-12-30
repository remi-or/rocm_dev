#include <hip/hip_runtime.h>
#include <iostream>

#define WARPSIZE 64
#define WARPTILE_M 8
#define WARPTILE_N 64

#define PT 3
#define CT 1
#define QSIZE 6

#define ELEMS_PER_CT ((WARPTILE_M * WARPTILE_N) / WARPSIZE)
#define CT_PER_ROW (WARPTILE_N / ELEMS_PER_CT)

void __global__ _tiled_sum_reduce_kernel(
    const float* __restrict__ A, 
    const float* __restrict__ B,
    float* __restrict__ D, 
    const int b,
    const int m,
    const int n
) {
    // Initialize shared queue
    __shared__ int queue[QSIZE];
    if (threadIdx.x == 0) {
        #pragma unroll
        for (int q = 0; q < QSIZE; q++) {
            queue[q] = 0;
        }
    }
    // Declare shared buffer
    __shared__ float A_buffer[WARPTILE_M * WARPTILE_N * QSIZE];
    __shared__ float B_buffer[WARPTILE_M * WARPTILE_N * QSIZE];
    __syncthreads();
    int buffer_offset, elems_per_thread, threads_per_row, curr_m, curr_n;

    // Determine warp specialization
    const int warp_id = threadIdx.x / WARPSIZE;
    const int thread_id = threadIdx.x % WARPSIZE;

    // Producer path
    if (warp_id < 2 * PT) {
        const float* __restrict__ src = (warp_id % 2 == 0) ? A : B;
        float* buffer = (warp_id % 2 == 0) ? &A_buffer[0] : &B_buffer[0];

        // Determine thread position
        elems_per_thread = (WARPTILE_M * WARPTILE_N) / (WARPSIZE); // = 4
        threads_per_row = WARPTILE_N / elems_per_thread; // = 16
        curr_m = (blockIdx.x * WARPTILE_M) + (thread_id / threads_per_row);
        curr_n = (blockIdx.y * WARPTILE_M) + (thread_id % threads_per_row) * elems_per_thread;
        // Relocate inputs
        src += (curr_m * n + curr_n) + m * n * (warp_id / 2);
        // Relocate buffer
        buffer += curr_m * WARPTILE_N + curr_n;

        // Batch-wise loop
        for (int curr_b = warp_id / 2; curr_b < b; curr_b+=PT) {

            // Wait for buffer to be consumed
            while (queue[curr_b % QSIZE] == 2) {
                asm volatile("s_sleep 0");
            }
            // Load
            buffer_offset = ((curr_b % QSIZE) * WARPTILE_M * WARPTILE_N);
            // #pragma unroll
            for (int i = 0; i < elems_per_thread; i++) {
                buffer[buffer_offset + i] = src[i];
            }
            // Advance
            src += (m * n) * PT;
            // Mark buffer as filled for this producer
            if (thread_id == 0) {
                atomicAdd(&queue[curr_b % QSIZE], 1);
            }
        }
    }

    // Consumers path
    else {

        // Determine thread position
        curr_m = (blockIdx.x * WARPTILE_M) + (thread_id / CT_PER_ROW);
        curr_n = (blockIdx.y * WARPTILE_M) + (thread_id % CT_PER_ROW) * CT_PER_ROW;

        // Initialize accumaltion registers
        float reg_D[ELEMS_PER_CT];
        #pragma unroll
        for (int i = 0; i < ELEMS_PER_CT; i++) {
            reg_D[i] = 0.0f;
        }

        // Batch-wise loop
        for (int curr_b = 0; curr_b < b; curr_b++) {

            // Wait for buffer to be filled
            while (queue[curr_b % QSIZE] != 2) {
                asm volatile("s_sleep 0");
            }
            // Accumulate
            buffer_offset = ((curr_b % QSIZE) * WARPTILE_M * WARPTILE_N) + (curr_m * WARPTILE_N + curr_n);
            #pragma unroll
            for (int i = 0; i < ELEMS_PER_CT; i++) {
                reg_D[i] += A_buffer[buffer_offset + i] + B_buffer[buffer_offset + i];
            }
            // Mark buffer as consummed
            if (thread_id == 0) {
                atomicSub(&queue[curr_b % QSIZE], 2);
            }
        }

        // Store
        D += curr_m * n + curr_n;
        #pragma unroll
        for (int i = 0; i < ELEMS_PER_CT; i++) {
            D[i] = reg_D[i];
        }
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
    dim3 block((2 * PT + CT) * WARPSIZE, 1, 1);

    // Launch kernel
    _tiled_sum_reduce_kernel<<<grid, block, 0, 0>>>(A, B, D, b, m, n);
}

