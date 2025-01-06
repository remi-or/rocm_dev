#include "./consumer.cu"
#include "./producer.cu"


void __global__ _tsr_kernel(
    const float* __restrict__ A, 
    const float* __restrict__ B,
    float* __restrict__ D, 
    const int m,
    const int n,
    const int k
) {
    // Initialize shared queue
    __shared__ uint8 queue[QSIZE];
    if (threadIdx.x == 0) {
        #pragma unroll
        for (int q = 0; q < 2 * QSIZE; q++) {
            queue[q] = 0;
        }
    }
    // Declare shared buffer
    __shared__ float A_buffer[WARPTILE_M * WARPTILE_K * QSIZE];
    __shared__ float B_buffer[WARPTILE_K * WARPTILE_N * QSIZE];
    __shared__ float D_buffer[(CONSUMERS == 1) ? 0 : (CONSUMERS * WARPTILE_M * WARPTILE_N)];
    __syncthreads();

    // Infer warp specialization
    const int warp_id = threadIdx.x / WARPSIZE;

    // Producer warp
    if (warp_id < (2 * PRODUCERS)) {

        // A producer
        if (warp_id % 2 == 0) {
            _tsr_producer<true>(A, &A_buffer[0], &queue[0], k);
        } 
        // B producer
        else {
            _tsr_producer<false>(B, &B_buffer[0], &queue[1], k);
        }
        
    }
    // Consumers warp
    else {
        uint16* q = reinterpret_cast<uint16*>(&queue[0]);
        _tsr_consumer(&A_buffer[0], &B_buffer[0], &D_buffer[0], D, q, n, k);
    }

    // If there is more than one consumer, we need to transfer the result from smem to gmem
    static constexpr int output_elems_per_thread = (WARPTILE_M * WARPTILE_N) / WARPSIZE;

    if (CONSUMERS > 1) {
        __syncthreads();
        if (warp_id == 0) {

            // Declare output registers
            float* d = &D_buffer[0] + threadIdx.x * output_elems_per_thread * CONSUMERS;
            float reg_D[output_elems_per_thread];

            // Loop through each output elements
            for (int i = 0; i < output_elems_per_thread; i++) {

                // Reduce across consumers
                reg_D[i] = 0.0f;
                #pragma unroll 
                for (int j = 0; j < CONSUMERS; j++) {
                    reg_D[i] += d[j];
                }
                d += CONSUMERS;
            }

            // Store back in gmem
            D += threadIdx.x * output_elems_per_thread;
            for (int i = 0; i < output_elems_per_thread; i++) {
                D[i] = reg_D[i];
            }
        }
    }
}

void async_gemm(
    const float* __restrict__ A, 
    const float* __restrict__ B,
    float* __restrict__ D, 
    const int m, 
    const int n, 
    const int k
) {
    // Check shapes
    if ((m % WARPTILE_M != 0) || (n % WARPTILE_N != 0) || (k % WARPTILE_K != 0)) {
        std::cerr << "Either m, n or k is not divisible by the corresponding WARPTILE_" << std::endl;
        exit(1);
    }

    // Prepare kernel launch
    const int grid_m = m / WARPTILE_M;
    const int grid_n = n / WARPTILE_N;
    dim3 grid(grid_m, grid_n, 1);
    dim3 block((2 * PRODUCERS + CONSUMERS) * WARPSIZE, 1, 1);

    // Launch kernel
    _tsr_kernel<<<grid, block, 0, 0>>>(A, B, D, m, n, k);
}
