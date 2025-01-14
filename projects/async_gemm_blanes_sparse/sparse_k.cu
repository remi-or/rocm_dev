#include "./consumer.cu"
#include "./producer.cu"

template <typename out_dtype>
void __global__ _tsr_kernel(
    const fp8* __restrict__ A, 
    const fp8* __restrict__ B,
    out_dtype* __restrict__ D, 
    const int m,
    const int n,
    const int k
) {
    // Initialize shared queue
    __shared__ uint8 queue[2 * B_LANES * QSIZE];
    if (threadIdx.x < 2 * B_LANES * QSIZE) {
        queue[threadIdx.x] = 0;
    }
    // Declare shared buffer
    __shared__ fp8 A_buffer[WARPTILE_M * WARPTILE_K * QSIZE];
    __shared__ fp8 B_buffer[WARPTILE_N * WARPTILE_K * QSIZE];
    __syncthreads();

    // Tiles loop
    int curr_n, curr_k, k_blocks;
    const int tiles = (n / WARPTILE_N) * SPLIT_K;
    const int tpw = max(CDIV(tiles, CU), 1);

    for (int warptile = (tpw * blockIdx.x); warptile < min(tiles, tpw * (blockIdx.x + 1)); warptile++) {

        // Compute tile position
        curr_n = (warptile % (n / WARPTILE_N)) * WARPTILE_N;
        curr_k = (warptile / (n / WARPTILE_N)) * WARPTILE_K * K_BLOCKS(k);
        k_blocks = ((warptile / (n / WARPTILE_N)) == SPLIT_K - 1) ? (k / WARPTILE_K) - (SPLIT_K - 1) * K_BLOCKS(k) : K_BLOCKS(k);

        // A producer warp
        if (threadIdx.x < A_PRODUCERS * WARPSIZE) {
            _tsr_A_producer(
                A + curr_k, 
                &A_buffer[0], 
                &queue[0],
                k, k_blocks
            ); 
        } 
        // B producer warp
        else if (threadIdx.x < A_PRODUCERS * WARPSIZE + B_LANES * B_PRODUCERS * WARPSIZE) {
            _tsr_B_producer(
                B + curr_n * k + curr_k,
                &B_buffer[0],
                &queue[0],
                k, k_blocks
            ); 
        }
        // Consumers warp
        else {
            _tsr_consumer(
                &A_buffer[0],
                &B_buffer[0],
                D + curr_n,
                &queue[0],
                n,
                k, k_blocks
            );
        }
    }    
}

template <typename out_dtype>
void async_gemm(
    const fp8* __restrict__ A, 
    const fp8* __restrict__ B,
    out_dtype* __restrict__ D, 
    const int m, 
    const int n, 
    const int k
) {
    // Check shapes
    if ((m % WARPTILE_M != 0) || (n % WARPTILE_N != 0) || (k % WARPTILE_K != 0)) {
        std::cerr << "Either m, n or k is not divisible by the corresponding WARPTILE_ :";
        std::cerr << m << ", " << n << ", " << k << std::endl;
        exit(1);
    }

    // Prepare kernel launch
    const int grid_m = m / WARPTILE_M;
    const int grid_n = n / WARPTILE_N;
    dim3 grid(CU, 1, 1);

    int warps = 0;
    warps += A_PRODUCERS;
    warps += B_PRODUCERS * B_LANES;
    warps += CONSUMERS * (TIED_CONSUMER ? B_LANES : 1);
    dim3 block(warps * WARPSIZE, 1, 1);

    // Launch kernel
    _tsr_kernel<<<grid, block, 0, 0>>>(A, B, D, m, n, k);
}




// void sparse_k(
//     torch::Tensor& A,
//     torch::Tensor& B,
//     torch::Tensor& D,
//     int64_t W
// ) {
//     const int m = A.size(0);
//     const int n = B.size(1);
//     const int k = A.size(1);
    
//     const fp8* __restrict__ A_ = (const fp8* __restrict__) A.data_ptr(); 
//     const fp8* __restrict__ B_ = (const fp8* __restrict__) B.data_ptr(); 
//     float* __restrict__ D_ = (float* __restrict__) D.data_ptr(); 

//         // Check shapes
//     if ((m % WARPTILE_M != 0) || (n % WARPTILE_N != 0) || (k % WARPTILE_K != 0)) {
//         std::cerr << "Either m, n or k is not divisible by the corresponding WARPTILE_ :";
//         std::cerr << m << ", " << n << ", " << k << std::endl;
//         exit(1);
//     }

//     // Prepare kernel launch
//     dim3 grid(CU, 1, 1);

//     int warps = 0;
//     warps += A_PRODUCERS;
//     warps += B_PRODUCERS * B_LANES;
//     warps += CONSUMERS * (TIED_CONSUMER ? B_LANES : 1);
//     dim3 block(warps * WARPSIZE, 1, 1);

//     const at::cuda::OptionalCUDAGuard device_guard(device_of(A));
//     const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

//     // Launch kernel
//     _tsr_kernel<float><<<grid, block, 0, stream>>>(A_, B_, D_, m, n, k);
// }
