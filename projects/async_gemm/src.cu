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
    __shared__ uint8 queue[2 * QSIZE]; // 00 00 00 00 00 00 = 0 / 257
    if (threadIdx.x == 0) {
        #pragma unroll
        for (int q = 0; q < 2 * QSIZE; q++) {
            queue[q] = 0;
        }
    }
    // Declare shared buffer
    __shared__ fp8 A_buffer[WARPTILE_M * WARPTILE_K * QSIZE];
    __shared__ fp8 B_buffer[WARPTILE_K * WARPTILE_N * QSIZE];
    __shared__ float D_buffer[(CONSUMERS > 1 && !G_ATOMICS) ? WARPTILE_M * WARPTILE_N * CONSUMERS : 0];
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
        _tsr_consumer(&A_buffer[0], &B_buffer[0], D, &D_buffer[0], q, n, k);
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
    if (OP_PER_WARPTILE != 1 && OP_PER_WARPTILE != 2) {
        std::cerr << "OP_PER_WARPTILE must be 1 or 2" << std::endl;
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
//     if ((m % WARPTILE_M != 0) || (n % WARPTILE_N != 0) || (k % WARPTILE_K != 0)) {
//         std::cerr << "Either m, n or k is not divisible by the corresponding WARPTILE_" << std::endl;
//         exit(1);
//     }
//     if (OP_PER_WARPTILE != 1 && OP_PER_WARPTILE != 2) {
//         std::cerr << "OP_PER_WARPTILE must be 1 or 2" << std::endl;
//         exit(1);
//     }

//     // Prepare kernel launch
//     uint32_t grid_m = (m / WARPTILE_M);
//     uint32_t grid_n = (n / WARPTILE_N);
//     dim3 grid(grid_m, grid_n, 1);
//     dim3 block((2 * PRODUCERS + CONSUMERS) * WARPSIZE, 1, 1);

//     const at::cuda::OptionalCUDAGuard device_guard(device_of(A));
//     const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

//     // Launch kernel
//     _tsr_kernel<float><<<grid, block, 0, stream>>>(A_, B_, D_, m, n, k);
// }
