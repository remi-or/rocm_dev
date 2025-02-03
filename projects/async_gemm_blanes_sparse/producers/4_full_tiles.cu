#include "./../core.cu" 

// WARNING / TODO : this producer always skips cache
template<int PRODUCERS, int B_LANES, int QSIZE, int OP_LD, int OP_AD>
void __device__ produce_4_full_tiles(
    const fp8* __restrict__ source,
    fp8* buffer,
    int* queue,
    int &index,
    int &p_state,
    int &role_id,
    const int k,
    const int k_blocks
) {
    static constexpr int E_per_thread = 16;
    static constexpr int E_per_bank = 4;
    static constexpr int Threads_per_ld = 4;
    static constexpr int Warptile_ld = 4 * OP_LD;

    // Infer ids
    const int thread_id = threadIdx.x % WARPSIZE;

    // Infer thread position in source
    const int curr_ld = (thread_id % Threads_per_ld) * E_per_thread;
    const int curr_ad = thread_id / Threads_per_ld;

    // Relocate thread in source
    source += curr_ad * k;
    source += curr_ld;

    // Relocate thread in buffer
    buffer += E_per_bank * curr_ad;
    buffer += (curr_ld / 32) * 32 * E_per_bank * 4;
    buffer += (curr_ld % 32 == 16) ? 16 * E_per_bank : 0;

    // Prepare registers
    fp8 reg0[E_per_thread];
    fp8 reg1[E_per_thread];
    fp8 reg2[E_per_thread];
    fp8 reg3[E_per_thread];

    // K-wise loop
    const fp8* src;
    fp8* buf;
    int b = role_id;

    while (b < B_LANES * k_blocks) {

        // Account for cyclic queue
        index -= (index >= QSIZE * B_LANES) ? QSIZE * B_LANES : 0;
        src = source + (b / B_LANES) * Warptile_ld + (b % B_LANES) * OP_AD * k;
        buf = buffer + index * (OP_AD * Warptile_ld);

        // Start loading all data
        asm volatile(
            "global_load_dwordx4 %0, %4, off offset:0   sc0 sc1 nt\n\t" 
            "global_load_dwordx4 %1, %4, off offset:64  sc0 sc1 nt\n\t" 
            "global_load_dwordx4 %2, %4, off offset:128 sc0 sc1 nt\n\t" 
            "global_load_dwordx4 %3, %4, off offset:192 sc0 sc1 nt\n\t"
            : "=v"(reg0), "=v"(reg1), "=v"(reg2), "=v"(reg3)
            : "v"(src)
        );

        // Wait for buffer to be consumed
        while (queue[2 * index] != p_state) {
            asm volatile("s_sleep 0");
        }

        // Fill 4-tile buffer
        asm volatile("s_waitcnt vmcnt(3)");
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            #pragma unroll
            for (int j = 0; j < 4; j++) {
                buf[NB_BANKS * E_per_bank * i + j] = reg0[E_per_bank * i + j];
            }
        }

        asm volatile("s_waitcnt vmcnt(2)");
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            #pragma unroll
            for (int j = 0; j < 4; j++) {
                buf[NB_BANKS * E_per_bank * i + j + OP_LD * OP_AD * 1] = reg1[E_per_bank * i + j];
            }
        }

        asm volatile("s_waitcnt vmcnt(1)");
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            #pragma unroll
            for (int j = 0; j < 4; j++) {
                buf[NB_BANKS * E_per_bank * i + j + OP_LD * OP_AD * 2] = reg2[E_per_bank * i + j];
            }
        }

        asm volatile("s_waitcnt vmcnt(0)");
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            #pragma unroll
            for (int j = 0; j < 4; j++) {
                buf[NB_BANKS * E_per_bank * i + j + OP_LD * OP_AD * 3] = reg3[E_per_bank * i + j];
            }
        }

        // Mark buffer as filled
        queue[2 * index] = p_state + 1;

        // Update loop variables
        index += PRODUCERS;
        p_state = (index >= QSIZE * B_LANES) ? (p_state + 2) : p_state;
        b += PRODUCERS;
    }

    // Bring warps back in order
    role_id = b - (B_LANES * k_blocks);
}
