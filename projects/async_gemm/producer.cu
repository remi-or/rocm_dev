#include "./core.cu"

template <bool A_producer>
void __device__ _tsr_producer(
    const fp8* __restrict__ src,
    fp8* buffer,
    uint8* queue,
    const int k
) {
    static constexpr int WARPTILE_MN = A_producer ? WARPTILE_M : WARPTILE_N;
    static constexpr int elems_per_thread = (WARPTILE_MN * WARPTILE_K) / WARPSIZE;
    static constexpr int threads_per_ld = WARPTILE_K / elems_per_thread;

    // Infer ids
    const int warp_id = threadIdx.x / WARPSIZE;
    const int thread_id = threadIdx.x % WARPSIZE;

    // Infer thread position in source
    const int blockIdx_xy = A_producer ? blockIdx.x : blockIdx.y;
    const int curr_ld = (thread_id % threads_per_ld) * elems_per_thread;
    const int curr_ad = thread_id / threads_per_ld;

    // Relocate thread in source
    src += (blockIdx_xy * WARPTILE_MN) * k + curr_ad * k;
    src += curr_ld;
    src += (warp_id / 2) * WARPTILE_K;

    // Relocate thread in buffer
    buffer += (curr_ld / OP_K) * (OP_MN * OP_K);
    const int buffer_ld = curr_ld % OP_K;
    buffer += (4 * curr_ad) + ((buffer_ld % 16 == 8) ? (16 * 4) : 0) + (buffer_ld / 16) * (32 * 4 * 2);

    // Prepare registers
    fp8 reg[elems_per_thread];

    // K-wise loop
    const int k_blocks = k / WARPTILE_K;
    int index;
    fp8* buf;

    for (int b = (warp_id / 2); b < k_blocks; b += PRODUCERS) {
        // Account for cyclic queue
        index = b % QSIZE;
        buf = buffer + index * (WARPTILE_MN * WARPTILE_K);

        // Load from gmem to reg
        #pragma unroll
        for (int i = 0; i < elems_per_thread; i++) {
            reg[i] = src[i];
        }

        // Wait for buffer to be consumed
        while (queue[2 * index] == 1) {
            asm volatile("s_sleep 0"); // TODO: try with 1
        }

        // Store in smem from reg // TODO: try with GMEM -> SMEM directly or with cache miss then hit
        buf[0] = reg[0];
        buf[1] = reg[1];
        buf[2] = reg[2];
        buf[3] = reg[3];

        buf[128] = reg[4];
        buf[129] = reg[5];
        buf[130] = reg[6];
        buf[131] = reg[7];

        if (OP_PER_WARPTILE == 2){
            buf[64] = reg[8];
            buf[65] = reg[9];
            buf[66] = reg[10];
            buf[67] = reg[11];

            buf[192] = reg[12];
            buf[193] = reg[13];
            buf[194] = reg[14];
            buf[195] = reg[15];
        }
        
        // Mark buffer as filled
        queue[2 * index] = 1;

        // Advance
        src += WARPTILE_K * PRODUCERS;
    }
}
