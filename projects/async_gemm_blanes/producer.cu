#include "./core.cu"

template <bool A_producer>
void __device__ _tsr_producer(
    const fp8* __restrict__ src,
    fp8* buffer,
    uint8* queue,
    const int k
) {
    static constexpr int WARPTILE_MN = A_producer ? WARPTILE_M : WARPTILE_N;
    static constexpr int elems_per_thread = (OP_MN * WARPTILE_K) / WARPSIZE;
    static constexpr int threads_per_ld = WARPTILE_K / elems_per_thread;

    // Infer ids
    const int warp_id = threadIdx.x / WARPSIZE;
    const int thread_id = threadIdx.x % WARPSIZE;
    const int warp_offs = A_producer ? warp_id : (warp_id - PRODUCERS) / B_LANES;

    // Infer thread position in source
    const int curr_ld = (thread_id % threads_per_ld) * elems_per_thread;
    const int curr_ad = thread_id / threads_per_ld;

    // Relocate thread in source (and queue for B producers)
    if (A_producer) {
        src += (blockIdx.x * WARPTILE_M + curr_ad) * k;
        src += curr_ld;
        src += warp_offs * WARPTILE_K;
    } else {
        src += (blockIdx.y * WARPTILE_N + curr_ad) * k;
        src += curr_ld;
        src += warp_offs * WARPTILE_K;
        src += ((warp_id - PRODUCERS) % B_LANES) * OP_MN * k;
        queue += 1 + 2 * ((warp_id - PRODUCERS) % B_LANES);
    }

    // Relocate thread in buffer
    buffer += (curr_ld / OP_K) * (OP_MN * OP_K);
    const int buffer_ld = curr_ld % OP_K;
    buffer += (4 * curr_ad) + ((buffer_ld % 16 == 8) ? (16 * 4) : 0) + (buffer_ld / 16) * (32 * 4 * 2);
    if (!A_producer) {
        buffer += ((warp_id - PRODUCERS) % B_LANES) * OP_MN * WARPTILE_K;
    }

    // Prepare registers
    fp8 reg[elems_per_thread];

    // K-wise loop
    const int k_blocks = infer_k_blocks(k);
    int index;
    fp8* buf;

    for (int b = warp_offs; b < k_blocks; b += PRODUCERS) {
        // Account for cyclic queue
        index = b % QSIZE;
        buf = buffer + index * (WARPTILE_MN * WARPTILE_K);

        // Load from gmem to reg
        #pragma unroll
        for (int i = 0; i < elems_per_thread; i++) {
            reg[i] = src[i];
        }

        // Wait for buffer to be consumed
        if (A_producer) {
            while (queue[2*B_LANES*index + 0] || queue[2*B_LANES*index + 2] || queue[2*B_LANES*index + 4]) {
                asm volatile("s_sleep 0"); // TODO: try with 1
            }
        } else {
            while (queue[2*B_LANES*index]) {
                asm volatile("s_sleep 0"); // TODO: try with 1
            }
        }

        // Store in smem from reg // TODO: try with GMEM -> SMEM directly or with cache miss then hit
        buf[0] = reg[0];
        buf[1] = reg[1];
        buf[2] = reg[2];
        buf[3] = reg[3];

        buf[128] = reg[4];
        buf[129] = reg[5];
        buf[130] = reg[6];
        buf[131] = reg[7];

        if (OP_PER_WARPTILE == 2){
            buf[64] = reg[8];
            buf[65] = reg[9];
            buf[66] = reg[10];
            buf[67] = reg[11];

            buf[192] = reg[12];
            buf[193] = reg[13];
            buf[194] = reg[14];
            buf[195] = reg[15];
        }
        
        // Mark buffer as filled
        if (A_producer) {
            queue[2 * B_LANES * index + 0] = 1;
            queue[2 * B_LANES * index + 2] = 1;
            queue[2 * B_LANES * index + 4] = 1;
        } else {
            queue[2 * B_LANES * index] = 1;
        }

        // Advance
        src += WARPTILE_K * PRODUCERS;
    }
}
