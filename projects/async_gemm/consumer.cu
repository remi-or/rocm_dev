#include "./core.cu"

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
    const int consumer_id = warp_id - 2 * PRODUCERS;
    const int warp_mask = 1 << (2 + consumer_id);

    // Initialize accumultion registers
    float reg_D[ELEMS_PER_THREADS];
    #pragma unroll
    for (int i = 0; i < ELEMS_PER_THREADS; i++) {
        reg_D[i] = 0.0f;
    }

    // Batch-wise loop
    int buffer_offset, old_q;
    for (int curr_b = consumer_id; curr_b < batches; curr_b+=CONSUMERS) {

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
            queue[curr_b % QSIZE] = 0;
        }
    }

    // Store in smem buffer
    D_buffer += consumer_id;
    #pragma unroll
    for (int i = 0; i < ELEMS_PER_THREADS; i++) {
        D_buffer[i * CONSUMERS] = reg_D[i];
    }
}
