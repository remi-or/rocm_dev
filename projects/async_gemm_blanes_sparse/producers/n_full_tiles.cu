#include "./../core.cu" 

// WARNING / TODO : this producer always skips cache
template<
    int PRODUCERS,
    int TILES,
    int LANES,
    int QSIZE,
    int OP_LEADING_SIZE,  // This is the size of the operation in the contiguous axis
    int OP_AUXILIARY_SIZE // This is the size of the operation in the strided axis
>
void __device__ produce_n_full_tiles(
    const fp8* __restrict__ source,
    fp8* buffer,
    int* queue,
    int q_stride,
    int &index,
    int &p_state,
    int &role_id,
    const int k,
    const int k_blocks
) {
    static constexpr int E_per_thread = 16;
    static constexpr int E_per_bank = 4;
    static constexpr int Threads_per_ld = 4;
    static constexpr int Warptile_ld = TILES * OP_LEADING_SIZE;

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
    fp8 reg[TILES][E_per_thread];

    // K-wise loop
    const fp8* src;
    fp8* buf;
    int b = role_id;

    while (b < LANES * k_blocks) {

        // Account for cyclic queue
        index -= (index >= QSIZE * LANES) ? QSIZE * LANES : 0;
        src = source + (b / LANES) * Warptile_ld + (b % LANES) * OP_AUXILIARY_SIZE * k;
        buf = buffer + index * (OP_AUXILIARY_SIZE * Warptile_ld);

        // Start loading all data
        if constexpr (TILES == 1) {
            asm volatile(
                "global_load_dwordx4 %0, %1, off offset:0   sc0 sc1 nt\n\t" 
                : "=v"(reg[0])
                : "v"(src)
            );
        }
        if constexpr (TILES == 2) {
            asm volatile(
                "global_load_dwordx4 %0, %2, off offset:0   sc0 sc1 nt\n\t" 
                "global_load_dwordx4 %1, %2, off offset:64  sc0 sc1 nt\n\t" 
                : "=v"(reg[0]), "=v"(reg[1])
                : "v"(src)
            );
        }
        if constexpr (TILES == 4) {
            asm volatile(
                "global_load_dwordx4 %0, %4, off offset:0   sc0 sc1 nt\n\t" 
                "global_load_dwordx4 %1, %4, off offset:64  sc0 sc1 nt\n\t" 
                "global_load_dwordx4 %2, %4, off offset:128 sc0 sc1 nt\n\t" 
                "global_load_dwordx4 %3, %4, off offset:192 sc0 sc1 nt\n\t"
                : "=v"(reg[0]), "=v"(reg[1]), "=v"(reg[2]), "=v"(reg[3])
                : "v"(src)
            );
        }

        // Wait for buffer to be consumed
        while (queue[2 * q_stride * index] != p_state) {
            asm volatile("s_sleep 0");
        }

        // Fill N-tile buffer
        #pragma unroll
        for (int tile = 0; tile < TILES; tile++) {

            // Wait for tile to be loaded in registers
            switch (TILES - tile - 1) {
                case 0: asm volatile("s_waitcnt vmcnt(0)"); break;
                case 1: asm volatile("s_waitcnt vmcnt(1)"); break;
                case 2: asm volatile("s_waitcnt vmcnt(2)"); break;
                case 3: asm volatile("s_waitcnt vmcnt(3)"); break;
            }

            // Place tile in shared memory
            #pragma unroll
            for (int i = 0; i < E_per_thread / E_per_bank; i++) {
                #pragma unroll
                for (int j = 0; j < E_per_bank; j++) {
                    buf[NB_BANKS * E_per_bank * i + j + OP_LEADING_SIZE * OP_AUXILIARY_SIZE * tile] = reg[tile][E_per_bank * i + j];
                }
            }
        }

        // Mark buffer as filled
        queue[2 * q_stride * index] = p_state + 1;

        // Update loop variables
        index += PRODUCERS;
        p_state = (index >= QSIZE * LANES) ? (p_state + 2) : p_state;
        b += PRODUCERS;
    }

    // Bring warps back in order
    role_id = b - (LANES * k_blocks);
}
