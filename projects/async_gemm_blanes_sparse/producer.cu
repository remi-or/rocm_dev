#include "./core.cu"

void __device__ _tsr_A_producer(
    const fp8* __restrict__ src,
    fp8* buffer,
    uint8* queue,
    int &index,
    uint8 &p_state,
    int &role_id,
    const int k,
    const int k_blocks
) {
    static constexpr int elems_per_thread = 16;
    static constexpr int threads_per_ld = 8;

    // Infer ids
    const int thread_id = threadIdx.x % WARPSIZE;

    // Infer thread position in source
    const int curr_ld = (thread_id % threads_per_ld) * elems_per_thread;
    const int curr_ad = thread_id / threads_per_ld;

    // Relocate thread in source (and queue for B producers) // WARNING: currently assume m == 8
    src += curr_ad * k;
    src += curr_ld;
    src += 2 * role_id * WARPTILE_K;

    // Relocate thread in buffer
    buffer += (E_P_BANK * curr_ad) + (curr_ld % 64) * 2 + (curr_ld / 64) * (32*E_P_BANK * 4);
    
    // Prepare registers
    fp8 reg[elems_per_thread];

    // K-wise loop
    fp8* buf;
    int b = 2 * role_id;

    while (b < k_blocks) {
        // Account for cyclic queue
        index -= (index >= QSIZE) ? QSIZE : 0;
        buf = buffer + index * (WARPTILE_M * WARPTILE_K);

        // Load from gmem to reg
        #pragma unroll
        for (int i = 0; i < elems_per_thread; i++) {
            reg[i] = src[i];
        }
        // Advance
        src += WARPTILE_K * 2*A_PRODUCERS;

        // LEFT --------------------------------------------------------------------------------------------------------
        // Wait for buffer to be consumed
        while (queue[2 * B_LANES * index] != p_state) {
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
        queue[2 * B_LANES * index] = 2;
        // Update index
        index += 1;

        // RIGHT -------------------------------------------------------------------------------------------------------
        // Wait for buffer to be consumed
        while (queue[2 * B_LANES * index] != p_state) {
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
        queue[2 * B_LANES * index] = 2;

        // Update index
        index += 2*A_PRODUCERS - 1;
        p_state = (index >= QSIZE) ? (!p_state) : p_state;
        b += 2*A_PRODUCERS;
    }

    // Bring warps back in order
    role_id = (b - k_blocks) / 2; // WARNING: not sure about this (and it's late)
}

void __device__ _tsr_B_producer(
    const fp8* __restrict__ source,
    fp8* buffer,
    uint8* queue,
    int &index,
    uint8 &p_state,
    int &role_id,
    const int k,
    const int k_blocks
) {
    static constexpr int elems_per_thread = 16;
    static constexpr int threads_per_ld = 4;

    // Infer ids
    const int thread_id = threadIdx.x % WARPSIZE;

    // Infer thread position in source
    const int curr_ld = (thread_id % threads_per_ld) * elems_per_thread;
    const int curr_ad = thread_id / threads_per_ld;

    // Relocate thread in source (and queue for B producers)
    source += curr_ad * k;
    source += curr_ld;

    // Relocate thread in buffer
    buffer += (E_P_BANK * curr_ad) + ((curr_ld % 32 == 16) ? (16*E_P_BANK) : 0) + (curr_ld / 32) * (32*E_P_BANK * 4);

    // Prepare registers
    fp8 reg[elems_per_thread];

    // K-wise loop
    const fp8* src;
    fp8* buf;
    int b = role_id;

    while (b < B_LANES * k_blocks) {

        // Account for cyclic queue
        index -= (index >= QSIZE * B_LANES) ? QSIZE * B_LANES : 0;
        src = source + (b / B_LANES) * WARPTILE_K + (b % B_LANES) * OP_N * k;
        buf = buffer + index * (OP_N * WARPTILE_K);
        // Load from gmem to reg
        asm volatile("global_load_dwordx4 %0, %1, off\n\t" : "=v"(reg) : "v"(src));
        // Wait for buffer to be consumed
        while (queue[2 * index] != p_state) {
            asm volatile("s_sleep 0");
        }
        // Make sure load is finished
        asm volatile("s_waitcnt vmcnt(0)");
        // Store in smem from reg
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            #pragma unroll
            for (int j = 0; j < 4; j++) {
                buf[32*4*i + j] = reg[4*i + j];
            }
        }
        // Mark buffer as filled
        queue[2 * index] = 2;

        // Update index
        index += B_PRODUCERS;
        p_state = (index >= QSIZE * B_LANES) ? (!p_state) : p_state;
        b += B_PRODUCERS;
    }

    // Bring warps back in order
    role_id = b - (B_LANES * k_blocks);
}
