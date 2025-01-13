#include "./core.cu"

void inline __device__ consumer_smem_to_reg(fp8* &buffer, fp8x8 &reg) 
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

void __device__ _tsr_consumer(
    fp8* A_buffer,
    fp8* B_buffer,
    float* D,
    uint16* queue,
    const int n,
    const int k
) {
    // Compile-time constants
    static constexpr int in_per_thread = (OP_MN * OP_K) / WARPSIZE;
    // Right now, this is hard coded to 4 for type issues # TODO: change back using asm instead of the builtin
    // static constexpr int out_per_thread = (WARPTILE_M * WARPTILE_N) / WARPSIZE;

    // Compute thread position
    const int thread_id = threadIdx.x % WARPSIZE;
    A_buffer += (thread_id % 32) * 4 + (thread_id / 32) * 256;
    B_buffer += (thread_id % 32) * 4 + (thread_id / 32) * 256;

    // Account for warp lane
    const int consumer_id = (threadIdx.x / WARPSIZE) - ((B_LANES + 1) * PRODUCERS);
    B_buffer += (consumer_id % B_LANES) * OP_MN * WARPTILE_K;
    queue += (consumer_id % B_LANES);
    D += (consumer_id % B_LANES) * OP_MN;

    // Declare input registers
    fp8x8 reg_A;
    fp8x8 reg_B;

    // Initialize output registers
    f32x4 reg_D;
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        reg_D[i] = 0.0f;
    }

    // K-wise loop
    int index;
    fp8 *A_offs_buff, *B_offs_buff;
    const int k_blocks = infer_k_blocks(k);

    for (int b = consumer_id / B_LANES; b < k_blocks; b += CONSUMERS) {
        index = b % QSIZE;
        A_offs_buff = A_buffer + index * (WARPTILE_M * WARPTILE_K);
        B_offs_buff = B_buffer + index * (WARPTILE_N * WARPTILE_K);

        // Wait for buffer to be filled
        while (queue[B_LANES * index] != PRODUCED_MASK) {
            asm volatile("s_sleep 0");
        }

        // Consume
        #pragma unroll
        for (int op = 0; op < OP_PER_WARPTILE; op++) {

            consumer_smem_to_reg(A_offs_buff, reg_A);
            consumer_smem_to_reg(B_offs_buff, reg_B);

            // Mark buffer as consumed
            if (op == (OP_PER_WARPTILE - 1)) {
                queue[B_LANES * index] = 0;
            }

            reg_D = __builtin_amdgcn_mfma_f32_16x16x32_fp8_fp8(
                reinterpret_cast<long>(reg_A),
                reinterpret_cast<long>(reg_B),
                reg_D, 0, 0, 0 // cbsz , abid, blgp
            );

            // Advance
            if (op < (OP_PER_WARPTILE - 1)) {
                A_offs_buff += OP_MN * OP_K;
                B_offs_buff += OP_MN * OP_K;
            }
        }
    }

    // Relocate on D
    int out_m = (thread_id / 16) * 4;
    int out_n = (thread_id % 16);
    D += (blockIdx.x * WARPTILE_M + out_m) * n + (blockIdx.y * WARPTILE_N + out_n);

    __syncthreads();

    int stride;
    float* out;

    // If there is only one consumer and no split-k, store directly in gmem
    if ((CONSUMERS == 1) && (SPLIT_K == 1)) {
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            D[i * n] = reg_D[i];
        }
    }

    // Otherwise, use global atomics
    else if (G_ATOMICS) {

        // Initialize if there is no split-k (otherwise, initializtion is assumed)
        if (SPLIT_K == 1) {
            if (consumer_id == 0) {
                #pragma unroll
                for (int i = 0; i < 4; i++) {
                    D[i * n] = reg_D[i];
                }
            }
            __syncthreads();
        }

        // Accumulate
        if (consumer_id > (1 - SPLIT_K)) {
            #pragma unroll
            for (int i = 0; i < 4; i++) {
                atomicAdd(&D[i * n], reg_D[i]);
            }
        }
    }

    // // Or shared buffer
    // else {
    //     // WIP
    // }
}
