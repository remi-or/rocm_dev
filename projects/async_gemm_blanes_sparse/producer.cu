#include "./core.cu"

void __device__ _tsr_A_producer(
    const fp8* __restrict__ src,
    fp8* buffer,
    uint8* queue,
    const int k
) {
    static constexpr int elems_per_thread = 16;
    static constexpr int threads_per_ld = 8;

    // Infer ids
    const int warp_id = threadIdx.x / WARPSIZE;
    const int thread_id = threadIdx.x % WARPSIZE;

    // Infer thread position in source
    const int curr_ld = ((thread_id % threads_per_ld) / 2) * elems_per_thread;
    const int curr_ad = thread_id / threads_per_ld;

    // Relocate thread in source (and queue for B producers)
    src += (blockIdx.x * WARPTILE_M + curr_ad) * k;
    src += curr_ld;
    src += warp_id * WARPTILE_K;

    // Relocate thread in buffer
    buffer += (2*E_P_BANK * curr_ad) + ((curr_ld % 32 == 16) ? (16*E_P_BANK) : 0) + (curr_ld / 32) * (32*E_P_BANK * 2);
    buffer += E_P_BANK * (threadIdx.x % 2);
    
    // Prepare registers
    fp8 reg[elems_per_thread];

    // K-wise loop
    const int k_blocks = infer_k_blocks(k);
    int index;
    fp8* buf;

    for (int b = warp_id; b < k_blocks; b += A_PRODUCERS) {
        // Account for cyclic queue
        index = b % QSIZE;
        buf = buffer + index * (WARPTILE_M * WARPTILE_K);

        // Load from gmem to reg
        #pragma unroll
        for (int i = 0; i < elems_per_thread; i++) {
            reg[i] = src[i];
        }

        // Wait for buffer to be consumed
        while (queue[2*B_LANES*index + 0] || queue[2*B_LANES*index + 2] || queue[2*B_LANES*index + 4]) {
            asm volatile("s_sleep 0"); // TODO: try with 1
        }

        // Store in smem from reg // TODO: try with GMEM -> SMEM directly or with cache miss then hit
        if (threadIdx.x % 2 == 0) {
            buf[0] = reg[0];
            buf[1] = reg[1];
            buf[2] = reg[4];
            buf[3] = reg[5];

            buf[32*E_P_BANK + 0] = reg[8];
            buf[32*E_P_BANK + 1] = reg[9];
            buf[32*E_P_BANK + 2] = reg[12];
            buf[32*E_P_BANK + 3] = reg[13];
        } else {
            buf[0] = reg[2];
            buf[1] = reg[3];
            buf[2] = reg[6];
            buf[3] = reg[7];

            buf[32*E_P_BANK + 0] = reg[10];
            buf[32*E_P_BANK + 1] = reg[11];
            buf[32*E_P_BANK + 2] = reg[14];
            buf[32*E_P_BANK + 3] = reg[15];
        }
        
        // Mark buffer as filled
        for (int l = 0; l < B_LANES; l++) {
            queue[2 * B_LANES * index + (2 * l)] = 1;
        }

        // Advance
        src += WARPTILE_K * A_PRODUCERS;
    }
}

void __device__ _tsr_B_producer(
    const fp8* __restrict__ src,
    fp8* buffer,
    uint8* queue,
    const int k
) {
    static constexpr int elems_per_thread = 16;
    static constexpr int threads_per_ld = 4;

    // Infer ids
    const int warp_id = threadIdx.x / WARPSIZE;
    const int thread_id = threadIdx.x % WARPSIZE;
    const int warp_offs = (warp_id - A_PRODUCERS) / B_LANES;
    const int warp_lane = (warp_id - A_PRODUCERS) % B_LANES;

    // Infer thread position in source
    const int curr_ld = (thread_id % threads_per_ld) * elems_per_thread;
    const int curr_ad = thread_id / threads_per_ld;

    // Relocate thread in source (and queue for B producers)
    src += (blockIdx.y * WARPTILE_N + curr_ad) * k;
    src += curr_ld;
    src += warp_offs * WARPTILE_K;
    src += warp_lane * OP_N * k;
    queue += 1 + 2 * warp_lane;

    // Relocate thread in buffer
    buffer += (E_P_BANK * curr_ad) + ((curr_ld % 32 == 16) ? (16*E_P_BANK) : 0) + (curr_ld / 32) * (32*E_P_BANK * 4);
    buffer += (warp_lane) * OP_N * WARPTILE_K;

    // Prepare registers
    fp8 reg[elems_per_thread];

    // K-wise loop
    const int k_blocks = infer_k_blocks(k);
    int index;
    fp8* buf;

    for (int b = warp_offs; b < k_blocks; b += B_PRODUCERS) {
        // Account for cyclic queue
        index = b % QSIZE;
        buf = buffer + index * (WARPTILE_N * WARPTILE_K);

        // Load from gmem to reg
        #pragma unroll
        for (int i = 0; i < elems_per_thread; i++) {
            reg[i] = src[i];
        }

        // Wait for buffer to be consumed
        while (queue[2*B_LANES*index]) {
            asm volatile("s_sleep 0"); // TODO: try with 1
        }

        // Store in smem from reg // TODO: try with GMEM -> SMEM directly or with cache miss then hit
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            #pragma unroll
            for (int j = 0; j < 4; j++) {
                buf[32*4*i + j] = reg[4*i + j];
            }
        }
        
        // Mark buffer as filled
        queue[2 * B_LANES * index] = 1;

        // Advance
        src += WARPTILE_K * B_PRODUCERS;
    }
}
