#include "./core.cu"

void inline __device__ consumer_smem_to_reg_8(fp8* &buffer, fp8x8 &reg) 
{
    // 32 bits load from the current bank
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        reg[i] = buffer[i];
    }
    // 32 bits load from the same bank, hopefully extension of the first load
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        reg[4 + i] = buffer[i + 4 * 32];
    }
}

void inline __device__ consumer_smem_to_reg_16(fp8* buffer, fp8x16 &reg) 
{
    #pragma unroll
    for (int i = 0; i < 4; i++) { reg[i     ] = buffer[i          ]; }
    #pragma unroll
    for (int i = 0; i < 4; i++) { reg[i +  4] = buffer[i +  4 * 32]; }
    #pragma unroll
    for (int i = 0; i < 4; i++) { reg[i +  8] = buffer[i +  8 * 32]; }
    #pragma unroll
    for (int i = 0; i < 4; i++) { reg[i + 12] = buffer[i + 12 * 32]; }
}

void __device__ _tsr_consumer(
    fp8* A_buffer,
    fp8* B_buffer,
    float* D,
    uint8* queue,
    const int n,
    const int k
) {
    // Compute thread position
    const int thread_id = threadIdx.x % WARPSIZE;
    A_buffer += (thread_id % 32) * 4 + (thread_id / 32) * 2 * E_PER_BANK * SMEM_BANKS;
    B_buffer += (thread_id % 32) * 4 + (thread_id / 32) * 4 * E_PER_BANK * SMEM_BANKS;

    // Account for warp lane
    const int consumer_id = (threadIdx.x / WARPSIZE) - 14;
    B_buffer += consumer_id * 2 * OP_N * WARPTILE_K;
    queue += 3 * consumer_id;
    D += 2 * consumer_id * OP_N;

    // Declare input registers
    fp8x8 reg_A;
    fp8x16 reg_B;

    // Initialize output registers
    f32x4 reg_D_left;
    f32x4 reg_D_right;
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        reg_D_left[i] = 0.0f;
        reg_D_right[i] = 0.0f;
    }

    // K-wise loop
    int index;
    fp8 *A_offs_buff, *B_offs_buff;
    const int k_blocks = infer_k_blocks(k);
    const int sparsity_indices = (threadIdx.x % 2) ? 0x0000EEEE : 0x00004444;

    for (int b = 0; b < k_blocks; b++) {
        index = (6 * b) % (3 * QSIZE);
        A_offs_buff = A_buffer + (b % (QSIZE / 2)) * (WARPTILE_M * OP_K);
        B_offs_buff = B_buffer + (b % (QSIZE / 2)) * (WARPTILE_N * OP_K);

        // Wait for buffer to be filled
        while (!queue[index] || !queue[index+1]) {
            asm volatile("s_sleep 0");
        }
        // Consume
        consumer_smem_to_reg_8(A_offs_buff, reg_A);
        queue[index] = 0;
        consumer_smem_to_reg_16(B_offs_buff, reg_B);
        queue[index+1] = 0;
        // Compute
        reg_D_left = __builtin_amdgcn_smfmac_f32_16x16x64_fp8_fp8(
            reinterpret_cast<fp8_4x2>(reg_A),
            reinterpret_cast<fp8_4x4>(reg_B),
            reg_D_left, 
            sparsity_indices, // src2
            7, 1 // cbsz, abid
        );

        // Wait for buffer to be filled
        while (!queue[index + 2]) {
            asm volatile("s_sleep 0");
        }
        // Consume
        B_offs_buff += OP_N * OP_K;
        consumer_smem_to_reg_16(B_offs_buff, reg_B);
        queue[index+2] = 0;
        // Compute
        reg_D_right = __builtin_amdgcn_smfmac_f32_16x16x64_fp8_fp8(
            reinterpret_cast<fp8_4x2>(reg_A),
            reinterpret_cast<fp8_4x4>(reg_B),
            reg_D_right, 
            sparsity_indices, // src2
            7, 1 // cbsz, abid
        );
    }

    // Fuse complementary registers
    reg_D_left[0] += reg_D_left[1];
    reg_D_left[2] += reg_D_left[3];
    reg_D_right[0] += reg_D_right[1];
    reg_D_right[2] += reg_D_right[3];

    // Relocate on D
    int out_m = (thread_id / 16) * 2;
    int out_n = (thread_id % 16);
    D += (blockIdx.x * WARPTILE_M + out_m) * n + (blockIdx.y * WARPTILE_N + out_n);

    __syncthreads();

    int stride;
    float* out;

    // If there is only one consumer and no split-k, store directly in gmem
    if ((CONSUMERS == 1) && (SPLIT_K == 1)) {
        #pragma unroll
        for (int i = 0; i < 2; i++) {
            D[i * n] = reg_D_left[2 * i];
            D[i * n + OP_N] = reg_D_right[2 * i];
        }
    }

    // Otherwise, use global atomics
    else if (G_ATOMICS) {

        // Initialize if there is no split-k (otherwise, initializtion is assumed)
        if (SPLIT_K == 1) {
            if (consumer_id == 0) {
                #pragma unroll
                for (int i = 0; i < 2; i++) {
                    D[i * n] = reg_D_left[2 * i];
                    D[i * n + OP_N] = reg_D_right[2 * i];
                }
            }
            __syncthreads();
        }

        // Accumulate
        if (consumer_id > (1 - SPLIT_K)) {
            #pragma unroll
            for (int i = 0; i < 2; i++) {
                atomicAdd(&D[i * n],reg_D_left[2 * i]);
                atomicAdd(&D[i * n + OP_N], reg_D_right[2 * i]);
            }
        }
    }

    // // Or shared buffer
    // else {
    //     // WIP
    // }
}
