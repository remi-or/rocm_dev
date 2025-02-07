#include "./../utils.cuh" 

// WARNING / TODO : this producer always skips cache
template<
    int PRODUCERS,
    int LANES,
    int QSIZE,
    int OP_MN, // This is the size of the operation in the strided axis
    bool REUSE
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
    static constexpr int E_PER_THREADS = 16;
    static constexpr int E_PER_BANK = 4;

    static constexpr int TILES_PER_LOADS = (WARPSIZE * E_PER_THREADS) / (OP_K * OP_MN); // = 32*16/16*64 = 2
    static constexpr int LOADS = OPS / TILES_PER_LOADS;

    static constexpr int Threads_per_ld = (OP_K * TILES_PER_LOADS) / E_PER_THREADS;

    // Infer ids
    const int lane_id = get_lane_id();

    // Infer thread position in source
    const int curr_ld = (lane_id % Threads_per_ld) * E_PER_THREADS;
    const int curr_ad = lane_id / Threads_per_ld;

    // Relocate thread in source
    source += curr_ad * k;
    source += curr_ld;

    // Relocate thread in buffer
    buffer += E_PER_BANK * curr_ad;
    buffer += (curr_ld / 32) * 32 * E_PER_BANK * 4;
    buffer += (curr_ld % 32 >= 16) ? 16 * E_PER_BANK : 0;

    // Prepare registers
    fp8 reg[LOADS][E_PER_THREADS];

    // K-wise loop
    const fp8* src;
    fp8* buf;
    int b = role_id;

    while (b < LANES * k_blocks) {

        // Account for cyclic queue
        index -= (index >= QSIZE * LANES) ? QSIZE * LANES : 0;
        src = source + (b / LANES) * WARPTILE_K + (b % LANES) * OP_MN * k;
        buf = buffer + index * (OP_MN * WARPTILE_K);

        // Start loading all data
        load_from_gmem_to_reg_no_waitcnt<LOADS, REUSE>(src, reg);

        // Wait for buffer to be consumed
        while (queue[2 * q_stride * index] != p_state) {
            asm volatile("s_sleep 0");
        }

        // Fill N-tile buffer
        #pragma unroll
        for (int load = 0; load < LOADS; load++) {

            // Wait for tile to be loaded in registers
            switch (LOADS - load - 1) {
                case 0: asm volatile("s_waitcnt vmcnt(0)"); break;
                case 1: asm volatile("s_waitcnt vmcnt(1)"); break;
                case 2: asm volatile("s_waitcnt vmcnt(2)"); break;
                case 3: asm volatile("s_waitcnt vmcnt(3)"); break;
            } // TODO: find a way to factor this out

            // Place tile in shared memory
            #pragma unroll
            for (int line = 0; line < E_PER_THREADS / E_PER_BANK; line++) {
                #pragma unroll
                for (int elem = 0; elem < E_PER_BANK; elem++) {
                    buf[line * (NB_BANKS * E_PER_BANK) + elem] = reg[load][E_PER_BANK * line + elem];
                }
            }
            buf += WARPSIZE * E_PER_THREADS;
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
