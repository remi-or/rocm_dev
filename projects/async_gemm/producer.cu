#include "./core.cu"

template<bool A_producer>
void __device__ _tsr_producer(
    const float* __restrict__ src,
    float* buffer,
    uint8* queue,
    const int k
) {
    static constexpr int WARPTILE_MN = A_producer ? WARPTILE_M : WARPTILE_N;
    static constexpr int elems_per_thread = (WARPTILE_MN * WARPTILE_K) / WARPSIZE;
    static constexpr int threads_per_ld = WARPTILE_K / elems_per_thread;

    const int warp_id = threadIdx.x / WARPSIZE;
    const int thread_id = threadIdx.x % WARPSIZE;

    // Relocate thread
    const int blockIdx_xy = A_producer ? blockIdx.x : blockIdx.y;
    src += (blockIdx_xy * WARPTILE_MN) * k + (thread_id / threads_per_ld) * k;
    src += (thread_id % threads_per_ld) * elems_per_thread;
    src += (warp_id / 2) * WARPTILE_K;

    buffer += (thread_id / threads_per_ld) * WARPTILE_K;
    buffer += (thread_id % threads_per_ld) * elems_per_thread;

    // K-wise loop
    const int k_blocks = k / WARPTILE_K;
    int index;
    float* buf;

    for (int b = (warp_id / 2); b < k_blocks; b += PRODUCERS) {
        // Account for cyclic queue
        index = b % QSIZE;

        // Wait for buffer to be consumed
        while (queue[2 * index] == 1) {
            asm volatile("s_sleep 0");
        }

        // Load
        buf = buffer + index * (WARPTILE_MN * WARPTILE_K);
        #pragma unroll
        for (int i = 0; i < elems_per_thread; i++) {
            buf[i] = src[i];
        }
        // Advance
        src += WARPTILE_K * PRODUCERS;

        // Mark buffer as filled
        if (thread_id == 0) {
            queue[2 * index] = 1;
        }
    }
}
