#include "./core.cu"

void __device__ _tsr_A_producer(
    const fp8* __restrict__ src,
    fp8* buffer,
    uint8* queue,
    const int k
) {
    static constexpr int elems_per_thread = 16; // = (OP_MN * WARPTILE_K) / WARPSIZE;
    static constexpr int threads_per_ld = 8;

    // Infer ids
    const int warp_id = threadIdx.x / WARPSIZE;
    const int thread_id = threadIdx.x % WARPSIZE;

    // Infer thread position in source
    const int curr_ld = (thread_id % threads_per_ld) * elems_per_thread;
    const int curr_ad = thread_id / threads_per_ld;

    // Relocate thread 
    const int buffer_ld = curr_ld % OP_K;

    src += (blockIdx.x * WARPTILE_M + curr_ad) * k;
    src += curr_ld;
    src += 2 * warp_id * WARPTILE_K;

    buffer += (2 * 4 * curr_ad) + ((buffer_ld % 32 == 16) ? (16 * 4) : 0) + (buffer_ld / 32) * (32 * 8);
    buffer += (curr_ld / OP_K) * (OP_M * OP_K);

    // Prepare registers
    fp8 reg[elems_per_thread];

    // K-wise loop
    const int k_blocks = infer_k_blocks(k);
    int index;
    fp8* buf;

    for (int b = 2 * warp_id; b < k_blocks; b+=(2 * A_PRODUCERS)) {
        // Account for cyclic queue
        index = (6 * b) % (3 * QSIZE);
        buf = buffer + (b % 6) * (OP_M * OP_K);

        // Load from gmem to reg
        #pragma unroll
        for (int i = 0; i < elems_per_thread; i++) {
            reg[i] = src[i];
        }

        // Wait for left buffer to be consumed
        while (queue[index] || queue[index + 3] || queue[index + 6] || queue[index + 9]) {
            asm volatile("s_sleep 0"); // TODO: try with 1
        }

        // Store in smem from reg // TODO: try with GMEM -> SMEM directly or with cache miss then hit
        buf[0] = reg[0];
        buf[1] = reg[1];
        buf[2] = reg[4];
        buf[3] = reg[5];

        buf[4] = reg[2];
        buf[5] = reg[3];
        buf[6] = reg[6];
        buf[7] = reg[7];

        buf[128] = reg[8];
        buf[129] = reg[9];
        buf[130] = reg[12];
        buf[131] = reg[13];

        buf[132] = reg[10];
        buf[133] = reg[11];
        buf[134] = reg[14];
        buf[135] = reg[15];
        
        // Mark buffer as filled
        queue[index + 0] = 1;
        queue[index + 3] = 1;
        queue[index + 6] = 1;
        queue[index + 9] = 1;

        // Advance
        src += 2 * OP_K * A_PRODUCERS;
    }
}


void __device__ _tsr_B_producer(
    const fp8* __restrict__ src,
    fp8* buffer,
    uint8* queue,
    const int k
) {
    static constexpr int elems_per_thread = 16;
    static constexpr int threads_per_ld = WARPTILE_K / elems_per_thread;

    // Infer ids
    const int warp_id = threadIdx.x / WARPSIZE;
    const int thread_id = threadIdx.x % WARPSIZE;
    const int warp_offs = (warp_id - A_PRODUCERS) / 4;
    const int warp_lane = (warp_id - A_PRODUCERS) % 4;

    // Infer thread position in source
    const int curr_ld = (thread_id % threads_per_ld) * elems_per_thread;
    const int curr_ad = thread_id / threads_per_ld;

    // Relocate thread 

    src += (blockIdx.y * WARPTILE_N + curr_ad) * k;
    src += curr_ld +  warp_offs * WARPTILE_K;
    src += warp_lane * OP_N * k;

    queue += 1 + warp_lane + warp_lane / 2;

    buffer += (4 * curr_ad) + ((curr_ld % 32 == 16) ? (16 * E_PER_BANK) : 0) + (curr_ld / 32) * (4 * E_PER_BANK * SMEM_BANKS);
    buffer += warp_lane * OP_N * WARPTILE_K;

    // Prepare registers
    fp8 reg[elems_per_thread];

    // K-wise loop
    const int k_blocks = infer_k_blocks(k);
    int index;
    fp8* buf;

    for (int b = warp_offs; b < k_blocks; b += B_PRODUCERS) {
        // Account for cyclic queue
        index = (6 * b) % (3 * QSIZE);
        buf = buffer + (b % (QSIZE / 2)) * (WARPTILE_N * OP_K);

        // Load from gmem to reg
        #pragma unroll
        for (int i = 0; i < elems_per_thread; i++) {
            reg[i] = src[i];
        }

        // Wait for buffer to be consumed
        while (queue[index]) {
            asm volatile("s_sleep 0"); // TODO: try with 1
        }
        
        // Store in smem from reg // TODO: try with GMEM -> SMEM directly or with cache miss then hit
        buf[0 * 4 * 32 + 0] = reg[0];
        buf[0 * 4 * 32 + 1] = reg[1];
        buf[0 * 4 * 32 + 2] = reg[2];
        buf[0 * 4 * 32 + 3] = reg[3];

        buf[1 * 4 * 32 + 0] = reg[4];
        buf[1 * 4 * 32 + 1] = reg[5];
        buf[1 * 4 * 32 + 2] = reg[6];
        buf[1 * 4 * 32 + 3] = reg[7];

        buf[2 * 4 * 32 + 0] = reg[8];
        buf[2 * 4 * 32 + 1] = reg[9];
        buf[2 * 4 * 32 + 2] = reg[10];
        buf[2 * 4 * 32 + 3] = reg[11];

        buf[3 * 4 * 32 + 0] = reg[12];
        buf[3 * 4 * 32 + 1] = reg[13];
        buf[3 * 4 * 32 + 2] = reg[14];
        buf[3 * 4 * 32 + 3] = reg[15];
        
        
        // Mark buffer as filled
        queue[index] = 1;

        // Advance
        src += WARPTILE_K * B_PRODUCERS;
    }
}
