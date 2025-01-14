#include "./core.cu"

void __device__ _tsr_A_producer(
    const fp8* __restrict__ src,
    fp8* buffer,
    uint8* queue,
    const int k,
    const int k_blocks
) {
    static constexpr int elems_per_thread = 16;
    static constexpr int threads_per_ld = 8;

    // Infer ids
    const int warp_id = threadIdx.x / WARPSIZE;
    const int thread_id = threadIdx.x % WARPSIZE;

    // Infer thread position in source
    const int curr_ld = (thread_id % threads_per_ld) * elems_per_thread;
    const int curr_ad = thread_id / threads_per_ld;

    // Relocate thread in source (and queue for B producers) // WARNING: currently assume m == 8
    src += curr_ad * k;
    src += curr_ld;
    src += 2 * warp_id * WARPTILE_K;

    // Relocate thread in buffer
    buffer += (E_P_BANK * curr_ad) + (curr_ld % 64) * 2 + (curr_ld / 64) * (32*E_P_BANK * 4);
    
    // Prepare registers
    fp8 reg[elems_per_thread];

    // K-wise loop
    int index;
    fp8* buf;

    for (int b = 2 * warp_id; b < k_blocks; b += 2*A_PRODUCERS) {
        // Account for cyclic queue
        index = b % QSIZE;
        buf = buffer + index * (WARPTILE_M * WARPTILE_K);

        // Load from gmem to reg
        #pragma unroll
        for (int i = 0; i < elems_per_thread; i++) {
            reg[i] = src[i];
        }

        // LEFT --------------------------------------------------------------------------------------------------------
        // Wait for buffer to be consumed
        while (queue[2*B_LANES*index + 0] || queue[2*B_LANES*index + 2] || queue[2*B_LANES*index + 4]) {
            asm volatile("s_sleep 0");
        }
        // Store in smem from reg
        if (threadIdx.x % 8 < 4) {
            #pragma unroll
            for (int line = 0; line < 4; line++) {
                #pragma unroll
                for (int j = 0; j < 4; j++){
                    buf[line * 32*E_P_BANK + j] = reg[8*(line%2) + 2*(line/2) + j + 2*(j/2)];
                }
            }
        }
        // Mark buffer as filled
        #pragma unroll
        for (int l = 0; l < B_LANES; l++) { queue[2 * B_LANES * index + (2 * l)] = 1; }

        // RIGHT -------------------------------------------------------------------------------------------------------
        // Wait for buffer to be consumed
        while (queue[2*B_LANES*index + 6] || queue[2*B_LANES*index + 8] || queue[2*B_LANES*index + 10]) {
            asm volatile("s_sleep 0");
        }
        // Store in smem from reg
        if (threadIdx.x % 8 > 3) {
            #pragma unroll
            for (int line = 0; line < 4; line++) {
                #pragma unroll
                for (int j = 0; j < 4; j++){
                    buf[line * 32*E_P_BANK + j] = reg[8*(line%2) + 2*(line/2) + j + 2*(j/2)];
                }
            }
        }
        // Mark buffer as filled
        #pragma unroll
        for (int l = 0; l < B_LANES; l++) { queue[2 * B_LANES * (index+1) + (2 * l)] = 1; }

        // Advance
        src += WARPTILE_K * 2*A_PRODUCERS;
    }
}

void __device__ _tsr_B_producer(
    const fp8* __restrict__ src,
    fp8* buffer,
    uint8* queue,
    const int k,
    const int k_blocks
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
    src += curr_ad * k;
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
