#include "./core.cu"

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
    for (int curr_b = (warp_id / 2); curr_b < batches; curr_b += PRODUCERS) {

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
        src += (m * n) * PRODUCERS;
        // Mark buffer as filled for this producer
        if (thread_id == 0) {
            queue[4 * (curr_b % QSIZE)] = 1;
        }
    }
}
