#include "./consumer.cu"
#include "./producer.cu"

template <typename T>
void __global__ _tsr_kernel(
    const fp8* __restrict__ A, 
    const fp8* __restrict__ B,
    T* __restrict__ D, 
    const int m,
    const int n,
    const int k,
    const int split_k
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


    // Infer index and p-state
    int role_id;
    int index;
    uint8 p_state;

    // A producer warp
    if (threadIdx.x < A_PRODUCERS * WARPSIZE) {
        role_id = threadIdx.x / WARPSIZE;
        index = (OPS == 2 ? 1 : 2) * role_id;
        p_state = 0;
    } 
    // B producer warp
    else if (threadIdx.x < A_PRODUCERS * WARPSIZE + B_PRODUCERS * WARPSIZE) {
        role_id = (threadIdx.x / WARPSIZE) - A_PRODUCERS;
        index = role_id;
        p_state = 0;
    }
    // Consumers warp
    else {
        role_id = (threadIdx.x / WARPSIZE) - (A_PRODUCERS + B_PRODUCERS);
        index = role_id;
        p_state = 1;
    }

    // Tiles loop
    int curr_n, curr_k, k_blocks, dropped_cols;
    const int warptile_per_row = CDIV(n, WARPTILE_N);
    const int tiles = warptile_per_row * split_k;
    const int tpw = max(CDIV(tiles, CU), 1);

    for (int warptile = (tpw * blockIdx.x); warptile < min(tiles, tpw * (blockIdx.x + 1)); warptile++) {

        // Compute tile position
        curr_n = (warptile % warptile_per_row) * WARPTILE_N;
        curr_k = (warptile / warptile_per_row) * WARPTILE_K * K_BLOCKS(k, split_k);
        k_blocks = ((warptile / warptile_per_row) == (split_k - 1)) ? (k / WARPTILE_K) - (split_k - 1) * K_BLOCKS(k, split_k) : K_BLOCKS(k, split_k);

        // Account for column overflow
        dropped_cols = max(0, curr_n + WARPTILE_N - n);
        curr_n -= dropped_cols;

        // A producer warp
        if (threadIdx.x < A_PRODUCERS * WARPSIZE) {
            _tsr_A_producer(
                A + curr_k, 
                &A_buffer[0], 
                &queue[0],
                index, p_state, role_id,
                k, k_blocks
            ); 
        } 
        // B producer warp
        else if (threadIdx.x < A_PRODUCERS * WARPSIZE + B_PRODUCERS * WARPSIZE) {
            _tsr_B_producer(
                B + curr_n * k + curr_k,
                &B_buffer[0],
                &queue[1],
                index, p_state, role_id,
                k, k_blocks
            ); 
        }
        // Consumers warp
        else {
            _tsr_consumer<T>(
                &A_buffer[0],
                &B_buffer[0],
                D + curr_n,
                &queue[0],
                index, p_state, role_id,
                n, dropped_cols,
                k, k_blocks
            );
        }
    }    
}

template <typename T>
void async_gemm(
    const fp8* __restrict__ A, 
    const fp8* __restrict__ B,
    T* __restrict__ D, 
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
    dim3 grid(CU, 1, 1);

    int warps = 0;
    warps += A_PRODUCERS;
    warps += B_PRODUCERS;
    warps += CONSUMERS;
    dim3 block(warps * WARPSIZE, 1, 1);

    // Launch kernel
    _tsr_kernel<T><<<grid, block, 0, 0>>>(A, B, D, m, n, k, SK);
}




// void sparse_k(
//     torch::Tensor& A,
//     torch::Tensor& B,
//     torch::Tensor& D,
//     int64_t split_k
// ) {
//     const int m = A.size(0);
//     const int n = B.size(1);
//     const int k = A.size(1);
    
//     const fp8* __restrict__ A_ = (const fp8* __restrict__) A.data_ptr(); 
//     const fp8* __restrict__ B_ = (const fp8* __restrict__) B.data_ptr(); 
//     half* __restrict__ D_ = (half* __restrict__) D.data_ptr(); 

//     // Check shapes
//     if ((m % WARPTILE_M != 0) || (k % WARPTILE_K != 0)) {
//         std::cerr << "Either m, n or k is not divisible by the corresponding WARPTILE_ :";
//         std::cerr << m << ", " << n << ", " << k << std::endl;
//         exit(1);
//     }

//     // Prepare kernel launch
//     dim3 grid(CU, 1, 1);

//     int warps = 0;
//     warps += A_PRODUCERS;
//     warps += B_PRODUCERS;
//     warps += CONSUMERS;
//     dim3 block(warps * WARPSIZE, 1, 1);

//     const at::cuda::OptionalCUDAGuard device_guard(device_of(A));
//     const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

//     // Launch kernel
//     _tsr_kernel<<<grid, block, 0, stream>>>(A_, B_, D_, m, n, k, split_k);
// }
