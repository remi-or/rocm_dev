#include "./../core.cu" 

// WARNING / TODO : tiles always hit cache AND are swizzled
template<int PRODUCERS, int B_LANES, int QSIZE>
void __device__ produce_4_half_tiles(
    const fp8* __restrict__ src,
    fp8* buffer,
    int* queue,
    int &index,
    int &p_state,
    int &role_id,
    const int dropped_ad,
    const int k,
    const int k_blocks
) {
    static constexpr int elems_per_thread = 16;
    static constexpr int elems_per_bank = 4;
    static constexpr int threads_per_ld = 8;

    // Infer ids
    const int thread_id = threadIdx.x % WARPSIZE;

    // Infer thread position in source
    const int curr_ld = (thread_id % threads_per_ld) * elems_per_thread;
    const int curr_ad = thread_id / threads_per_ld;

    // Relocate thread in source (and queue for B producers) // WARNING: currently assume m == 8
    src += (curr_ad + dropped_ad > WARPTILE_M - 1) ? 0 : curr_ad * k;
    src += curr_ld;
    src += (OPS == 1 ? 2 : 1) * role_id * WARPTILE_K;

    // Relocate thread in buffer
    buffer += elems_per_bank * curr_ad;
    buffer += (curr_ld / 64) * 32 * elems_per_bank * 4;
    buffer += (curr_ld % 64) * 2;
    
    // Prepare registers
    fp8_4 reg0[elems_per_thread / 4];
    fp8_4 reg1[elems_per_thread / 4];
    fp8_4 swap_reg[elems_per_thread / 4];

    // K-wise loop
    fp8_4* buf;
    int b = (OPS == 1 ? 2 : 1) * role_id;

    while (b < k_blocks) {

        // Account for cyclic queue
        index -= (index >= QSIZE) ? QSIZE : 0;
        buf = reinterpret_cast<fp8_4*>(buffer) + index * (WARPTILE_M * WARPTILE_K / 4);

        // Load from gmem to reg
        asm volatile(
            "global_load_dwordx4 %0, %2, off offset:0  \n\t"
            "global_load_dwordx4 %1, %2, off offset:128\n\t"
            : "=v"(reg0), "=v"(reg1)
            : "v"(src)
        );

        // Wait for buffer to be consumed
        while (queue[2 * B_LANES * index] != p_state) {
            asm volatile("s_sleep 0");
        }

        // Store in smem from reg
        asm volatile(
            "s_waitcnt vmcnt(1)\n\t"
            "v_pack_b32_f16 %0, %4, %5, op_sel:[0,0]\n\t"
            "v_pack_b32_f16 %1, %4, %5, op_sel:[1,1]\n\t"
            "v_pack_b32_f16 %2, %6, %7, op_sel:[0,0]\n\t"
            "v_pack_b32_f16 %3, %6, %7, op_sel:[1,1]\n\t"
            "v_swap_b32 %0, %4\n\t"
            "v_swap_b32 %1, %6\n\t"
            "v_swap_b32 %2, %5\n\t"
            "v_swap_b32 %3, %7"
            : "=v"(swap_reg[0]), "=v"(swap_reg[1]), "=v"(swap_reg[2]), "=v"(swap_reg[3])
            : "v"(reg0[0]), "v"(reg0[1]), "v"(reg0[2]), "v"(reg0[3])
        );
        #pragma unroll
        for (int line = 0; line < 4; line++) {
            buf[line * 32] = reg0[line];
        }

        // Store in smem from reg
        asm volatile(
            "s_waitcnt vmcnt(0)\n\t"
            "v_pack_b32_f16 %0, %4, %5, op_sel:[0,0]\n\t"
            "v_pack_b32_f16 %1, %4, %5, op_sel:[1,1]\n\t"
            "v_pack_b32_f16 %2, %6, %7, op_sel:[0,0]\n\t"
            "v_pack_b32_f16 %3, %6, %7, op_sel:[1,1]\n\t"
            "v_swap_b32 %0, %4\n\t"
            "v_swap_b32 %1, %6\n\t"
            "v_swap_b32 %2, %5\n\t"
            "v_swap_b32 %3, %7"
            : "=v"(swap_reg[0]), "=v"(swap_reg[1]), "=v"(swap_reg[2]), "=v"(swap_reg[3])
            : "v"(reg1[0]), "v"(reg1[1]), "v"(reg1[2]), "v"(reg1[3])
        );
        #pragma unroll
        for (int line = 0; line < 4; line++) {
            buf[line * 32 + OP_K * OP_M / 2] = reg1[line];
        }

        // Mark buffer as filled
        queue[2 * B_LANES * index] = p_state + 1;

        // Advance
        src += WARPTILE_K * PRODUCERS;

        // Update index
        index += PRODUCERS;
        p_state = (index >= QSIZE) ? (p_state + 2) : p_state;
        b += PRODUCERS;
    }

    // Bring warps back in order
    role_id = (b - k_blocks); // WARNING: not sure about this (and it's late)
}
