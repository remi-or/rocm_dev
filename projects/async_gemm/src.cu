#include "./consumer.cu"
#include "./producer.cu"


void __global__ _tsr_kernel(
    const float* __restrict__ A, 
    const float* __restrict__ B,
    float* __restrict__ D, 
    const int b,
    const int m,
    const int n
) {
    // Initialize shared queue
    __shared__ unsigned int queue[QSIZE];
    if (threadIdx.x == 0) {
        #pragma unroll
        for (int q = 0; q < QSIZE; q++) {
            queue[q] = 0;
        }
    }
    // Declare shared buffer
    __shared__ float A_buffer[WARPTILE_M * WARPTILE_N * QSIZE];
    __shared__ float B_buffer[WARPTILE_M * WARPTILE_N * QSIZE];
    __shared__ float D_buffer[(CONSUMERS == 1) ? 1 : (CONSUMERS * WARPTILE_M * WARPTILE_N)];
    __syncthreads();

    // Determine warp specialization
    const int warp_id = threadIdx.x / WARPSIZE;
    const int thread_id = threadIdx.x % WARPSIZE;
    // Determine thread position
    int curr_m = (blockIdx.x * WARPTILE_M) + (thread_id / THREADS_PER_ROW);
    int curr_n = (blockIdx.y * WARPTILE_M) + (thread_id % THREADS_PER_ROW) * ELEMS_PER_THREADS;

    // Producer path
    if (warp_id < (2 * PRODUCERS)) {
        const float* __restrict__ src = (warp_id % 2 == 0) ? A : B;
        float* buffer = (warp_id % 2 == 0) ? &A_buffer[0] : &B_buffer[0];
        uint8* q = reinterpret_cast<uint8*>(&queue[0]) + (warp_id % 2);

        _tsr_producer(
            src + (curr_m * n + curr_n) + (warp_id / 2) * m * n,
            buffer + (curr_m * WARPTILE_N + curr_n),
            q,
            warp_id, thread_id,
            b, m, n
        );
    }

    // Consumers path
    else {
        float* out = (CONSUMERS == 1) ? D : &D_buffer[0];
        _tsr_consumer(
            &A_buffer[0] + (curr_m * WARPTILE_N + curr_n),
            &B_buffer[0] + (curr_m * WARPTILE_N + curr_n),
            out + (curr_m * WARPTILE_N + curr_n) * CONSUMERS,
            &queue[0],
            warp_id,
            thread_id,
            b
        );
    }
    __syncthreads();

    // Final reduce and store
    float results_reg[ELEMS_PER_THREADS];
    if ((CONSUMERS > 1) && (warp_id == 0)) {
        
        #pragma unroll
        for (int i = 0; i < ELEMS_PER_THREADS; i++) {
            results_reg[i] = 0.0f;

            #pragma unroll 
            for (int j = 0; j < CONSUMERS; j++) {
                results_reg[i] += D_buffer[(curr_m * WARPTILE_N + curr_n + i) * CONSUMERS + j];
            }
        }

        D += curr_m * n + curr_n;
        for (int i = 0; i < ELEMS_PER_THREADS; i++) {
            D[i] = results_reg[i];
        }
    }
}

void tiled_sum_reduce(
    const float* __restrict__ A, 
    const float* __restrict__ B,
    float* __restrict__ D, 
    const int b, 
    const int m, 
    const int n
) {
    // Check shapes
    if ((m % WARPTILE_M != 0) || (n % WARPTILE_N != 0)) {
        std::cerr << "Either m or n is not divisible by the corresponding WARPTILE_" << std::endl;
    }

    // Prepare kernel launch
    const int grid_m = m / WARPTILE_M;
    const int grid_n = n / WARPTILE_N;
    dim3 grid(grid_m, grid_n, 1);
    dim3 block((2 * PRODUCERS + CONSUMERS) * WARPSIZE, 1, 1);

    // Launch kernel
    _tsr_kernel<<<grid, block, 0, 0>>>(A, B, D, b, m, n);
}
