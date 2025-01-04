#include <hip/hip_runtime.h>
#include <iostream>

#define WARPSIZE 64
#define WARPTILE_M 8
#define WARPTILE_N 64

#define PT 4
#define CT 1
#define QSIZE 8
#define PRODUCED_MASK 257
#define FINISHED_MASK ((1 << (2 + CT)) - 1)

#define ELEMS_PER_THREADS ((WARPTILE_M * WARPTILE_N) / WARPSIZE)
#define THREADS_PER_ROW (WARPTILE_N / ELEMS_PER_THREADS)

using uint8 = unsigned char;

void inline __device__ _tsr_producer(
    const float* __restrict__ src,
    float* buffer,
    uint8* queue,
    const int warp_id,
    const int thread_id,
    const int batches,
    const int m,
    const int n
) { 
    // Batch-wise loop
    int buffer_offset;
    for (int curr_b = (warp_id / 2); curr_b < batches; curr_b += PT) {

        // Wait for buffer to be consumed
        while (queue[4 * (curr_b % QSIZE)] == 1) {
            asm volatile("s_sleep 0");
        }
        // Load
        buffer_offset = ((curr_b % QSIZE) * WARPTILE_M * WARPTILE_N);
        #pragma unroll
        for (int i = 0; i < ELEMS_PER_THREADS; i++) {
            buffer[buffer_offset + i] = src[i];
        }
        // Advance
        src += (m * n) * PT;
        // Mark buffer as filled for this producer
        if (thread_id == 0) {
            queue[4 * (curr_b % QSIZE)] = 1;
        }
    }
}

void inline __device__ _tsr_consumer(
    float* A_buffer,
    float* B_buffer,
    float* D_buffer,
    unsigned int* queue,
    const int warp_id,
    const int thread_id,
    const int batches
) {
    // Determine warp position
    const int consumer_id = warp_id - 2 * PT;
    const int warp_mask = 1 << (2 + consumer_id);

    // Initialize accumultion registers
    float reg_D[ELEMS_PER_THREADS];
    #pragma unroll
    for (int i = 0; i < ELEMS_PER_THREADS; i++) {
        reg_D[i] = 0.0f;
    }

    // Batch-wise loop
    int buffer_offset, old_q;
    for (int curr_b = consumer_id; curr_b < batches; curr_b+=CT) {

        // Wait for buffer to be filled
        while (queue[curr_b % QSIZE] < PRODUCED_MASK) {
            asm volatile("s_sleep 0");
        }
        // Accumulate
        buffer_offset = ((curr_b % QSIZE) * WARPTILE_M * WARPTILE_N);
        #pragma unroll
        for (int i = 0; i < ELEMS_PER_THREADS; i++) {
            reg_D[i] += A_buffer[buffer_offset + i] + B_buffer[buffer_offset + i];
        }
        // Mark buffer as consumed
        if (thread_id == 0) {
            if (CT == 1) {
                queue[curr_b % QSIZE] = 0;
            } else {
                old_q = atomicInc(&queue[curr_b % QSIZE], warp_mask);
                if (old_q == CT - 1) {
                    queue[curr_b % QSIZE] = 0;
                }
            }
        }
    }

    // Store in smem buffer
    D_buffer += consumer_id;
    #pragma unroll
    for (int i = 0; i < ELEMS_PER_THREADS; i++) {
        D_buffer[i * CT] = reg_D[i];
    }
}

void __global__ _tsr_kernel(
    const float* __restrict__ A, 
    const float* __restrict__ B,
    float* __restrict__ D, 
    const int b,
    const int m,
    const int n
) {
    // Initialize shared queue
    __shared__ unsigned int queue[QSIZE];
    if (threadIdx.x == 0) {
        #pragma unroll
        for (int q = 0; q < QSIZE; q++) {
            queue[q] = 0;
        }
    }
    // Declare shared buffer
    __shared__ float A_buffer[WARPTILE_M * WARPTILE_N * QSIZE];
    __shared__ float B_buffer[WARPTILE_M * WARPTILE_N * QSIZE];
    __shared__ float D_buffer[CT * WARPTILE_M * WARPTILE_N];
    __syncthreads();
    int warp_offset;

    // Determine warp specialization
    const int warp_id = threadIdx.x / WARPSIZE;
    const int thread_id = threadIdx.x % WARPSIZE;
    // Determine thread position
    int curr_m = (blockIdx.x * WARPTILE_M) + (thread_id / THREADS_PER_ROW);
    int curr_n = (blockIdx.y * WARPTILE_M) + (thread_id % THREADS_PER_ROW) * ELEMS_PER_THREADS;

    // Producer path
    if (warp_id < (2 * PT)) {
        const float* __restrict__ src = (warp_id % 2 == 0) ? A : B;
        float* buffer = (warp_id % 2 == 0) ? &A_buffer[0] : &B_buffer[0];
        uint8* q = reinterpret_cast<uint8*>(&queue[0]) + (warp_id % 2);

        _tsr_producer(
            src + (curr_m * n + curr_n) + (warp_id / 2) * m * n,
            buffer + (curr_m * WARPTILE_N + curr_n),
            q,
            warp_id, thread_id,
            b, m, n
        );
    }

    // Consumers path
    else {
        _tsr_consumer(
            &A_buffer[0] + (curr_m * WARPTILE_N + curr_n),
            &B_buffer[0] + (curr_m * WARPTILE_N + curr_n),
            &D_buffer[0] + (curr_m * WARPTILE_N + curr_n) * CT,
            &queue[0],
            warp_id,
            thread_id,
            b
        );
    }
    __syncthreads();

    // Final reduce and store
    float results_reg[ELEMS_PER_THREADS];
    if (warp_id == 0) {
        
        #pragma unroll
        for (int i = 0; i < ELEMS_PER_THREADS; i++) {
            results_reg[i] = 0.0f;

            #pragma unroll 
            for (int j = 0; j < CT; j++) {
                results_reg[i] += D_buffer[(curr_m * WARPTILE_N + curr_n + i) * CT + j];
            }
        }

        D += curr_m * n + curr_n;
        for (int i = 0; i < ELEMS_PER_THREADS; i++) {
            D[i] = results_reg[i];
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
    _tsr_kernel<<<grid, block, 0, 0>>>(A, B, D, b, m, n);
}
