#include "./core.cu"

void inline __device__ consumer_smem_to_reg8(fp8* &buffer, fp8x8 &reg) 
{
    // 32 bits load from the current bank
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        reg[i] = buffer[i];
    }
    // 32 bits load from the same bank, hopefully extension of the first load
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        reg[4 + i] = buffer[i + 32*E_P_BANK];
    }
}

void inline __device__ consumer_smem_to_reg16(fp8* &buffer, fp8x16 &reg) 
{
    #pragma unroll
    for (int i = 0; i < 4; i++) { reg[i     ] = buffer[i                   ]; }
    #pragma unroll
    for (int i = 0; i < 4; i++) { reg[i +  4] = buffer[i +     32*E_P_BANK]; }
    #pragma unroll
    for (int i = 0; i < 4; i++) { reg[i +  8] = buffer[i + 2 * 32*E_P_BANK]; }
    #pragma unroll
    for (int i = 0; i < 4; i++) { reg[i + 12] = buffer[i + 3 * 32*E_P_BANK]; }
}

void __device__ _tsr_consumer(
    fp8* A_buffer,
    fp8* B_buffer,
    float* D,
    uint8* queue,
    const int n,
    const int k,
    const int k_blocks
) {
    // Compute thread position
    const int thread_id = threadIdx.x % WARPSIZE;
    A_buffer += (thread_id / 2) * E_P_BANK + (threadIdx.x % 2) * 32*E_P_BANK * 2;
    B_buffer += (thread_id % 32) * 4 + (thread_id / 32) * 32*E_P_BANK * 4;
    const int sparsity_indices = (threadIdx.x % 2) ? 0x0000EEEE : 0x00004444;

    // Account for warp lane
    const int consumer_id = (threadIdx.x / WARPSIZE) - (A_PRODUCERS + B_PRODUCERS);

    // Declare input registers
    fp8x8 reg_A;
    fp8x16 reg_B;

    // Initialize output registers
    f32x4 reg_D[B_LANES];
    #pragma unroll
    for (int i = 0; i < (B_LANES); i++) {
        reg_D[i][0] = 0.0f; reg_D[i][1] = 0.0f; reg_D[i][2] = 0.0f; reg_D[i][3] = 0.0f;
    }

    // K-wise loop
    uint8 p_state = 1;
    int index = consumer_id;
    fp8 *A_offs_buff, *B_offs_buff;

    for (int b = consumer_id; b < k_blocks; b+=CONSUMERS) {

        // Account for cyclic queue
        A_offs_buff = A_buffer + index * (WARPTILE_M * WARPTILE_K);
        B_offs_buff = B_buffer + index * (WARPTILE_N * WARPTILE_K);

        // Load A buffer
        while (!queue[2 * B_LANES * index]) {
            asm volatile("s_sleep 0");
        }
        consumer_smem_to_reg8(A_offs_buff, reg_A);
        queue[2 * B_LANES * index] = 0;

        #pragma unroll
        for (int lane = 0; lane < B_LANES; lane++) {

            // Wait for buffer to be filled
            while (queue[2 * (B_LANES * index + lane) + 1] != 2) {
                asm volatile("s_sleep 0");
            }

            // Consume
            consumer_smem_to_reg16(B_offs_buff, reg_B);
            // Mark buffer as consumed
            queue[2 * (B_LANES * index + lane) + 1] = p_state;

            reg_D[lane] = __builtin_amdgcn_smfmac_f32_16x16x64_fp8_fp8(
                reinterpret_cast<fp8_4x2>(reg_A),
                reinterpret_cast<fp8_4x4>(reg_B),
                reg_D[lane], 
                sparsity_indices, // src2
                7, 1 // cbsz, abid
            );

            B_offs_buff += OP_N * OP_K;
        }

        // Update index
        index += CONSUMERS;
        p_state = (index >= QSIZE) ? (!p_state) : p_state;
        index -= (index >= QSIZE) ? QSIZE : 0;
    }
    __syncthreads();
    // if (threadIdx.x == 1023) {
    //     for (int i = 0; i < B_LANES * QSIZE; i++) {
    //         queue[2*i+1] = 0;
    //     }
    // }
    // __syncthreads();

    // Fuse complementary registers
    #pragma unroll
    for (int i = 0; i < B_LANES; i++) {
        reg_D[i][0] += reg_D[i][1];
        reg_D[i][2] += reg_D[i][3];
    }

    // Relocate on D
    int out_m = (thread_id / 16) * 2;
    int out_n = (thread_id % 16);
    D += out_m * n + out_n;


    // If there is only one consumer and no split-k, store directly in gmem
    if ((CONSUMERS == 1) && (SPLIT_K == 1)) {
        #pragma unroll
        for (int i = 0; i < B_LANES; i++) {
            D[    i*OP_N] = reg_D[i][0];
            D[n + i*OP_N] = reg_D[i][2];
        }
    }

    // Otherwise, use global atomics
    else if (G_ATOMICS) {

        // Initialize if there is no split-k (otherwise, initializtion is assumed)
        // if (SPLIT_K == 1) {
        //     if (consumer_id == 0) {
        //         #pragma unroll
        //         for (int i = 0; i < (TIED_CONSUMER ? 1 : B_LANES); i++) {
        //             D[0 + i*OP_N] = reg_D[i][0];
        //             D[n + i*OP_N] = reg_D[i][2];
        //         }
        //     }
        //     __syncthreads();
        // }

        // Accumulate
        // if (consumer_id > (1 - SPLIT_K)) {
        #pragma unroll
        for (int i = 0; i < B_LANES; i++) {
            atomicAdd(&D[0 + i*OP_N], reg_D[i][0]);
            atomicAdd(&D[n + i*OP_N], reg_D[i][2]);
        }
        // }
    }

    // // Or shared buffer
    // else {
    //     // WIP
    // }
}
