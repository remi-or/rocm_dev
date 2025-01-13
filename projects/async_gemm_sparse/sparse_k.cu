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
    __shared__ uint8 queue[3 * QSIZE];
    if (threadIdx.x == 0) {
        #pragma unroll
        for (int q = 0; q < 3 * QSIZE; q++) {
            queue[q] = 0;
        }
    }
    // Declare shared buffer
    __shared__ fp8 A_buffer[WARPTILE_M * WARPTILE_K * (QSIZE / 2)];
    __shared__ fp8 B_buffer[WARPTILE_N * WARPTILE_K * (QSIZE / 2)];
    __syncthreads();

    // Infer warp specialization
    const int warp_id = threadIdx.x / WARPSIZE;

    // Account for split-k
    A += blockIdx.z * WARPTILE_K * K_BLOCKS(k);
    B += blockIdx.z * WARPTILE_K * K_BLOCKS(k);

    // A producer warp
    if (warp_id < 2) {
        _tsr_A_producer(A, &A_buffer[0], &queue[0], k); } 
    // B producer warp
    else if (warp_id < 14) {
        _tsr_B_producer(B, &B_buffer[0], &queue[0], k); }
    // Consumers warp
    else {
        _tsr_consumer(&A_buffer[0], &B_buffer[0], D, &queue[0], n, k);
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
    if ((m % WARPTILE_M != 0) || (k % WARPTILE_K != 0)) {
        std::cerr << "Either m, n or k is not divisible by the corresponding WARPTILE_ :";
        std::cerr << m << ", " << n << ", " << k << std::endl;
        exit(1);
    }

    // Prepare kernel launch
    const int grid_m = m / WARPTILE_M;
    const int grid_n = n / WARPTILE_N;
    dim3 grid(grid_m, grid_n, SPLIT_K);
    dim3 block(16 * WARPSIZE, 1, 1);

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

//     // Check shapes
//     if ((m % WARPTILE_M != 0) || (k % WARPTILE_K != 0)) {
//         std::cerr << "Either m, n or k is not divisible by the corresponding WARPTILE_ :";
//         std::cerr << m << ", " << n << ", " << k << std::endl;
//         exit(1);
//     }

//     // Prepare kernel launch
//     const int grid_m = m / WARPTILE_M;
//     const int grid_n = n / WARPTILE_N;
//     dim3 grid(grid_m, grid_n, SPLIT_K);
//     dim3 block(16 * WARPSIZE, 1, 1);

//     const at::cuda::OptionalCUDAGuard device_guard(device_of(A));
//     const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

//     // Launch kernel
//     _tsr_kernel<float><<<grid, block, 0, stream>>>(A_, B_, D_, m, n, k);
// }
