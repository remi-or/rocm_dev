#include "./core.cu"

void inline __device__ consume(
    float* A_buffer,
    float* B_buffer,
    float* reg_D,
    const int curr_m,
    const int curr_n,
    int elems_per_thread,
    int threads_per_row
) {
    // Relocate thread
    float* a = A_buffer + curr_m * WARPTILE_K;
    float* b;

    for (int i = 0; i < elems_per_thread; i++) {
        b = B_buffer + (curr_n + i) * WARPTILE_K;

        #pragma unroll
        for (int j = 0; j < WARPTILE_K; j++) {
            reg_D[i] += a[j] * b[j];
        }
    }
}

void __device__ _tsr_consumer(
    float* A_buffer,
    float* B_buffer,
    float* D_buffer,
    float* D,
    uint16* queue,
    const int n,
    const int k
) {
    static constexpr int elems_per_thread = (WARPTILE_M * WARPTILE_N) / WARPSIZE;
    static constexpr int threads_per_row = WARPTILE_N / elems_per_thread;

    // Compute thread position
    const int thread_id = threadIdx.x % WARPSIZE;
    const int curr_m = thread_id / threads_per_row;
    const int curr_n = (thread_id % threads_per_row) * elems_per_thread;

    // Initialize accumultion registers
    float reg_D[elems_per_thread];
    #pragma unroll
    for (int i = 0; i < elems_per_thread; i++) {
        reg_D[i] = 0.0f;
    }

    // K-wise loop
    int index;
    const int consumer_id = (threadIdx.x / WARPSIZE) - (2 * PRODUCERS);
    const int k_blocks = k / WARPTILE_K;

    for (int b = consumer_id; b < k_blocks; b += CONSUMERS) {
        index = b % QSIZE;

        // Wait for buffer to be filled
        while (queue[index] != PRODUCED_MASK) {
            asm volatile("s_sleep 0");
        }
        // Consume 
        consume(
            A_buffer + index * (WARPTILE_M * WARPTILE_K),
            B_buffer + index * (WARPTILE_N * WARPTILE_K),
            reg_D,
            curr_m, 
            curr_n,
            elems_per_thread,
            threads_per_row
        );

        // Mark buffer as consumed
        if (thread_id == 0) {
            queue[index] = 0;
        }
    }

    // If there is only one consumer, store directly in gmem
    float* out;
    if (CONSUMERS == 1) {
        out = D + (blockIdx.x * WARPTILE_M + curr_m) * n + (blockIdx.y * WARPTILE_N + curr_n); }
    // Otherwise, store in a shared memory buffer
    else {
        out = D_buffer + ((curr_m * WARPTILE_M) + curr_n) * CONSUMERS + consumer_id;
    }

    // Store
    #pragma unroll
    for (int i = 0; i < elems_per_thread; i++) {
        out[i * CONSUMERS] = reg_D[i];
    }
}
