#include "./skinny_gemm_kernel.cu"

// #define LAUNCH_ONE_SKINNY_GEMM(__bl, __qs, __opm, __ops) _skinny_gemm_kernel                      \
//     <__bl, __qs, __opm, __ops>                                                                  \
//     <<<grid, block, 0, stream>>>                                                              \
//     (A, B, D, scale_tensor, m, n, k, b_stride, split_k, A_producers, B_producers, consumers);

#define COND_LAUCNH_ONE_SKINNY_GEMM(__bl, __qs, __om, __ops)                                         \
    else if (b_lanes == __bl && qsize == __qs && op_m == __om && ops == __ops) {                     \
        _skinny_gemm_kernel<__bl, __qs, __om, __ops><<<grid, block, 0, stream>>>(                    \
            A, B, D, scale_tensor, m, n, k, b_stride, split_k, A_producers, B_producers, consumers); \
    }

int skinny_gemm(
// Tensors
    const fp8* __restrict__ A,
    const fp8* __restrict__ B,
    half* __restrict__ D,
    const float* scale_tensor,
// Shapes
    const int m,
    const int n,
    const int k,
    const int b_stride,
    const int split_k,
// Async non-templated
    const int A_producers,
    const int B_producers,
    const int consumers,
// Async templated
    const int b_lanes,
    const int qsize,
    const int op_m,
    const int ops,
// Cuda-related
    hipStream_t stream
) {

    // Deduce other constants
    const int OP_K = 512 / op_m;
    const int WARPTILE_M = op_m;
    const int WARPTILE_K = OP_K * ops;

    // Check shapes
    if (m > WARPTILE_M) {
        std::cerr << "m = " << m << " is greater than WARPTILE_M = " << WARPTILE_M << std::endl;
        return 1;
    }
    if (n % 2 != 0) {
        std::cerr << "n = " << n << " is not even" << std::endl;
        return 2;
    }
    if (k % WARPTILE_K != 0) {
        std::cerr << "k = " << k << " is not divisible by WARPTILE_K = " << WARPTILE_K << std::endl;
        return 3;
    }

    // Check async
    if (A_producers + B_producers + consumers > 16) {
        std::cerr << "A_producers = " << A_producers << ", B_producers = " << B_producers << ", consumers = ";
        std::cerr << consumers << " is greater than 16" << std::endl;
        return 4;
    }
    if (qsize < A_producers || qsize < (B_producers / b_lanes) || qsize < consumers) {
        std::cerr << "qsize = " << qsize << " is less than A_producers = " << A_producers;
        std::cerr << ", B_producers / b_lanes = " << B_producers / b_lanes;
        std::cerr << ", or consumers = " << consumers << std::endl;
        return 5;
    }

    // Prepare kernel launch
    dim3 grid(CU);
    dim3 block((A_producers + B_producers + consumers) * WARPSIZE);

    // Dispatch to the correct kernel
    if (b_lanes == 0) { return ; }
    COND_LAUCNH_ONE_SKINNY_GEMM(1, 1, 8, 2)
    COND_LAUCNH_ONE_SKINNY_GEMM(1, 1, 8, 4)
    COND_LAUCNH_ONE_SKINNY_GEMM(1, 1, 8, 8)
    COND_LAUCNH_ONE_SKINNY_GEMM(1, 1, 16, 2)
    COND_LAUCNH_ONE_SKINNY_GEMM(1, 1, 16, 4)
    COND_LAUCNH_ONE_SKINNY_GEMM(1, 1, 16, 8)
    COND_LAUCNH_ONE_SKINNY_GEMM(1, 1, 32, 2)
    COND_LAUCNH_ONE_SKINNY_GEMM(1, 1, 32, 4)
    COND_LAUCNH_ONE_SKINNY_GEMM(1, 1, 32, 8)
    COND_LAUCNH_ONE_SKINNY_GEMM(1, 2, 8, 2)
    COND_LAUCNH_ONE_SKINNY_GEMM(1, 2, 8, 4)
    COND_LAUCNH_ONE_SKINNY_GEMM(1, 2, 8, 8)
    COND_LAUCNH_ONE_SKINNY_GEMM(1, 2, 16, 2)
    COND_LAUCNH_ONE_SKINNY_GEMM(1, 2, 16, 4)
    COND_LAUCNH_ONE_SKINNY_GEMM(1, 2, 16, 8)
    COND_LAUCNH_ONE_SKINNY_GEMM(1, 2, 32, 2)
    COND_LAUCNH_ONE_SKINNY_GEMM(1, 2, 32, 4)
    COND_LAUCNH_ONE_SKINNY_GEMM(1, 2, 32, 8)
    COND_LAUCNH_ONE_SKINNY_GEMM(1, 3, 8, 2)
    COND_LAUCNH_ONE_SKINNY_GEMM(1, 3, 8, 4)
    COND_LAUCNH_ONE_SKINNY_GEMM(1, 3, 8, 8)
    COND_LAUCNH_ONE_SKINNY_GEMM(1, 3, 16, 2)
    COND_LAUCNH_ONE_SKINNY_GEMM(1, 3, 16, 4)
    COND_LAUCNH_ONE_SKINNY_GEMM(1, 3, 16, 8)
    COND_LAUCNH_ONE_SKINNY_GEMM(1, 3, 32, 2)
    COND_LAUCNH_ONE_SKINNY_GEMM(1, 3, 32, 4)
    COND_LAUCNH_ONE_SKINNY_GEMM(1, 3, 32, 8)
    COND_LAUCNH_ONE_SKINNY_GEMM(1, 4, 8, 2)
    COND_LAUCNH_ONE_SKINNY_GEMM(1, 4, 8, 4)
    COND_LAUCNH_ONE_SKINNY_GEMM(1, 4, 8, 8)
    COND_LAUCNH_ONE_SKINNY_GEMM(1, 4, 16, 2)
    COND_LAUCNH_ONE_SKINNY_GEMM(1, 4, 16, 4)
    COND_LAUCNH_ONE_SKINNY_GEMM(1, 4, 16, 8)
    COND_LAUCNH_ONE_SKINNY_GEMM(1, 4, 32, 2)
    COND_LAUCNH_ONE_SKINNY_GEMM(1, 4, 32, 4)
    COND_LAUCNH_ONE_SKINNY_GEMM(1, 4, 32, 8)
    COND_LAUCNH_ONE_SKINNY_GEMM(1, 5, 8, 2)
    COND_LAUCNH_ONE_SKINNY_GEMM(1, 5, 8, 4)
    COND_LAUCNH_ONE_SKINNY_GEMM(1, 5, 8, 8)
    COND_LAUCNH_ONE_SKINNY_GEMM(1, 5, 16, 2)
    COND_LAUCNH_ONE_SKINNY_GEMM(1, 5, 16, 4)
    COND_LAUCNH_ONE_SKINNY_GEMM(1, 5, 16, 8)
    COND_LAUCNH_ONE_SKINNY_GEMM(1, 5, 32, 2)
    COND_LAUCNH_ONE_SKINNY_GEMM(1, 5, 32, 4)
    COND_LAUCNH_ONE_SKINNY_GEMM(1, 5, 32, 8)
    COND_LAUCNH_ONE_SKINNY_GEMM(1, 6, 8, 2)
    COND_LAUCNH_ONE_SKINNY_GEMM(1, 6, 8, 4)
    COND_LAUCNH_ONE_SKINNY_GEMM(1, 6, 16, 2)
    COND_LAUCNH_ONE_SKINNY_GEMM(1, 6, 16, 4)
    COND_LAUCNH_ONE_SKINNY_GEMM(1, 6, 16, 8)
    COND_LAUCNH_ONE_SKINNY_GEMM(1, 6, 32, 2)
    COND_LAUCNH_ONE_SKINNY_GEMM(1, 6, 32, 4)
    COND_LAUCNH_ONE_SKINNY_GEMM(1, 6, 32, 8)
    COND_LAUCNH_ONE_SKINNY_GEMM(2, 1, 8, 2)
    COND_LAUCNH_ONE_SKINNY_GEMM(2, 1, 8, 4)
    COND_LAUCNH_ONE_SKINNY_GEMM(2, 1, 8, 8)
    COND_LAUCNH_ONE_SKINNY_GEMM(2, 1, 16, 2)
    COND_LAUCNH_ONE_SKINNY_GEMM(2, 1, 16, 4)
    COND_LAUCNH_ONE_SKINNY_GEMM(2, 1, 16, 8)
    COND_LAUCNH_ONE_SKINNY_GEMM(2, 1, 32, 2)
    COND_LAUCNH_ONE_SKINNY_GEMM(2, 1, 32, 4)
    COND_LAUCNH_ONE_SKINNY_GEMM(2, 1, 32, 8)
    COND_LAUCNH_ONE_SKINNY_GEMM(2, 2, 8, 2)
    COND_LAUCNH_ONE_SKINNY_GEMM(2, 2, 8, 4)
    COND_LAUCNH_ONE_SKINNY_GEMM(2, 2, 8, 8)
    COND_LAUCNH_ONE_SKINNY_GEMM(2, 2, 16, 2)
    COND_LAUCNH_ONE_SKINNY_GEMM(2, 2, 16, 4)
    COND_LAUCNH_ONE_SKINNY_GEMM(2, 2, 16, 8)
    COND_LAUCNH_ONE_SKINNY_GEMM(2, 2, 32, 2)
    COND_LAUCNH_ONE_SKINNY_GEMM(2, 2, 32, 4)
    COND_LAUCNH_ONE_SKINNY_GEMM(2, 2, 32, 8)
    COND_LAUCNH_ONE_SKINNY_GEMM(2, 3, 8, 2)
    COND_LAUCNH_ONE_SKINNY_GEMM(2, 3, 8, 4)
    COND_LAUCNH_ONE_SKINNY_GEMM(2, 3, 8, 8)
    COND_LAUCNH_ONE_SKINNY_GEMM(2, 3, 16, 2)
    COND_LAUCNH_ONE_SKINNY_GEMM(2, 3, 16, 4)
    COND_LAUCNH_ONE_SKINNY_GEMM(2, 3, 16, 8)
    COND_LAUCNH_ONE_SKINNY_GEMM(2, 3, 32, 2)
    COND_LAUCNH_ONE_SKINNY_GEMM(2, 3, 32, 4)
    COND_LAUCNH_ONE_SKINNY_GEMM(2, 3, 32, 8)
    COND_LAUCNH_ONE_SKINNY_GEMM(2, 4, 8, 2)
    COND_LAUCNH_ONE_SKINNY_GEMM(2, 4, 8, 4)
    COND_LAUCNH_ONE_SKINNY_GEMM(2, 4, 16, 2)
    COND_LAUCNH_ONE_SKINNY_GEMM(2, 4, 16, 4)
    COND_LAUCNH_ONE_SKINNY_GEMM(2, 4, 16, 8)
    COND_LAUCNH_ONE_SKINNY_GEMM(2, 4, 32, 2)
    COND_LAUCNH_ONE_SKINNY_GEMM(2, 4, 32, 4)
    COND_LAUCNH_ONE_SKINNY_GEMM(2, 4, 32, 8)
    COND_LAUCNH_ONE_SKINNY_GEMM(2, 5, 8, 2)
    COND_LAUCNH_ONE_SKINNY_GEMM(2, 5, 8, 4)
    COND_LAUCNH_ONE_SKINNY_GEMM(2, 5, 16, 2)
    COND_LAUCNH_ONE_SKINNY_GEMM(2, 5, 16, 4)
    COND_LAUCNH_ONE_SKINNY_GEMM(2, 5, 16, 8)
    COND_LAUCNH_ONE_SKINNY_GEMM(2, 5, 32, 2)
    COND_LAUCNH_ONE_SKINNY_GEMM(2, 5, 32, 4)
    COND_LAUCNH_ONE_SKINNY_GEMM(2, 5, 32, 8)
    COND_LAUCNH_ONE_SKINNY_GEMM(2, 6, 8, 2)
    COND_LAUCNH_ONE_SKINNY_GEMM(2, 6, 8, 4)
    COND_LAUCNH_ONE_SKINNY_GEMM(2, 6, 16, 2)
    COND_LAUCNH_ONE_SKINNY_GEMM(2, 6, 16, 4)
    COND_LAUCNH_ONE_SKINNY_GEMM(2, 6, 32, 2)
    COND_LAUCNH_ONE_SKINNY_GEMM(2, 6, 32, 4)
    COND_LAUCNH_ONE_SKINNY_GEMM(3, 1, 8, 2)
    COND_LAUCNH_ONE_SKINNY_GEMM(3, 1, 8, 4)
    COND_LAUCNH_ONE_SKINNY_GEMM(3, 1, 8, 8)
    COND_LAUCNH_ONE_SKINNY_GEMM(3, 1, 16, 2)
    COND_LAUCNH_ONE_SKINNY_GEMM(3, 1, 16, 4)
    COND_LAUCNH_ONE_SKINNY_GEMM(3, 1, 16, 8)
    COND_LAUCNH_ONE_SKINNY_GEMM(3, 1, 32, 2)
    COND_LAUCNH_ONE_SKINNY_GEMM(3, 1, 32, 4)
    COND_LAUCNH_ONE_SKINNY_GEMM(3, 1, 32, 8)
    COND_LAUCNH_ONE_SKINNY_GEMM(3, 2, 8, 2)
    COND_LAUCNH_ONE_SKINNY_GEMM(3, 2, 8, 4)
    COND_LAUCNH_ONE_SKINNY_GEMM(3, 2, 8, 8)
    COND_LAUCNH_ONE_SKINNY_GEMM(3, 2, 16, 2)
    COND_LAUCNH_ONE_SKINNY_GEMM(3, 2, 16, 4)
    COND_LAUCNH_ONE_SKINNY_GEMM(3, 2, 16, 8)
    COND_LAUCNH_ONE_SKINNY_GEMM(3, 2, 32, 2)
    COND_LAUCNH_ONE_SKINNY_GEMM(3, 2, 32, 4)
    COND_LAUCNH_ONE_SKINNY_GEMM(3, 2, 32, 8)
    COND_LAUCNH_ONE_SKINNY_GEMM(3, 3, 8, 2)
    COND_LAUCNH_ONE_SKINNY_GEMM(3, 3, 8, 4)
    COND_LAUCNH_ONE_SKINNY_GEMM(3, 3, 16, 2)
    COND_LAUCNH_ONE_SKINNY_GEMM(3, 3, 16, 4)
    COND_LAUCNH_ONE_SKINNY_GEMM(3, 3, 16, 8)
    COND_LAUCNH_ONE_SKINNY_GEMM(3, 3, 32, 2)
    COND_LAUCNH_ONE_SKINNY_GEMM(3, 3, 32, 4)
    COND_LAUCNH_ONE_SKINNY_GEMM(3, 3, 32, 8)
    COND_LAUCNH_ONE_SKINNY_GEMM(3, 4, 8, 2)
    COND_LAUCNH_ONE_SKINNY_GEMM(3, 4, 8, 4)
    COND_LAUCNH_ONE_SKINNY_GEMM(3, 4, 16, 2)
    COND_LAUCNH_ONE_SKINNY_GEMM(3, 4, 16, 4)
    COND_LAUCNH_ONE_SKINNY_GEMM(3, 4, 32, 2)
    COND_LAUCNH_ONE_SKINNY_GEMM(3, 4, 32, 4)
    COND_LAUCNH_ONE_SKINNY_GEMM(3, 5, 8, 2)
    COND_LAUCNH_ONE_SKINNY_GEMM(3, 5, 16, 2)
    COND_LAUCNH_ONE_SKINNY_GEMM(3, 5, 16, 4)
    COND_LAUCNH_ONE_SKINNY_GEMM(3, 5, 32, 2)
    COND_LAUCNH_ONE_SKINNY_GEMM(3, 5, 32, 4)
    COND_LAUCNH_ONE_SKINNY_GEMM(3, 6, 8, 2)
    COND_LAUCNH_ONE_SKINNY_GEMM(3, 6, 16, 2)
    COND_LAUCNH_ONE_SKINNY_GEMM(3, 6, 16, 4)
    COND_LAUCNH_ONE_SKINNY_GEMM(3, 6, 32, 2)
    COND_LAUCNH_ONE_SKINNY_GEMM(3, 6, 32, 4)
    COND_LAUCNH_ONE_SKINNY_GEMM(4, 1, 8, 2)
    COND_LAUCNH_ONE_SKINNY_GEMM(4, 1, 8, 4)
    COND_LAUCNH_ONE_SKINNY_GEMM(4, 1, 8, 8)
    COND_LAUCNH_ONE_SKINNY_GEMM(4, 1, 16, 2)
    COND_LAUCNH_ONE_SKINNY_GEMM(4, 1, 16, 4)
    COND_LAUCNH_ONE_SKINNY_GEMM(4, 1, 16, 8)
    COND_LAUCNH_ONE_SKINNY_GEMM(4, 1, 32, 2)
    COND_LAUCNH_ONE_SKINNY_GEMM(4, 1, 32, 4)
    COND_LAUCNH_ONE_SKINNY_GEMM(4, 1, 32, 8)
    COND_LAUCNH_ONE_SKINNY_GEMM(4, 2, 8, 2)
    COND_LAUCNH_ONE_SKINNY_GEMM(4, 2, 8, 4)
    COND_LAUCNH_ONE_SKINNY_GEMM(4, 2, 16, 2)
    COND_LAUCNH_ONE_SKINNY_GEMM(4, 2, 16, 4)
    COND_LAUCNH_ONE_SKINNY_GEMM(4, 2, 16, 8)
    COND_LAUCNH_ONE_SKINNY_GEMM(4, 2, 32, 2)
    COND_LAUCNH_ONE_SKINNY_GEMM(4, 2, 32, 4)
    COND_LAUCNH_ONE_SKINNY_GEMM(4, 2, 32, 8)
    COND_LAUCNH_ONE_SKINNY_GEMM(4, 3, 8, 2)
    COND_LAUCNH_ONE_SKINNY_GEMM(4, 3, 8, 4)
    COND_LAUCNH_ONE_SKINNY_GEMM(4, 3, 16, 2)
    COND_LAUCNH_ONE_SKINNY_GEMM(4, 3, 16, 4)
    COND_LAUCNH_ONE_SKINNY_GEMM(4, 3, 16, 8)
    COND_LAUCNH_ONE_SKINNY_GEMM(4, 3, 32, 2)
    COND_LAUCNH_ONE_SKINNY_GEMM(4, 3, 32, 4)
    COND_LAUCNH_ONE_SKINNY_GEMM(4, 3, 32, 8)
    COND_LAUCNH_ONE_SKINNY_GEMM(4, 4, 8, 2)
    COND_LAUCNH_ONE_SKINNY_GEMM(4, 4, 16, 2)
    COND_LAUCNH_ONE_SKINNY_GEMM(4, 4, 16, 4)
    COND_LAUCNH_ONE_SKINNY_GEMM(4, 4, 32, 2)
    COND_LAUCNH_ONE_SKINNY_GEMM(4, 4, 32, 4)
    COND_LAUCNH_ONE_SKINNY_GEMM(4, 5, 8, 2)
    COND_LAUCNH_ONE_SKINNY_GEMM(4, 5, 16, 2)
    COND_LAUCNH_ONE_SKINNY_GEMM(4, 5, 16, 4)
    COND_LAUCNH_ONE_SKINNY_GEMM(4, 5, 32, 2)
    COND_LAUCNH_ONE_SKINNY_GEMM(4, 5, 32, 4)
    COND_LAUCNH_ONE_SKINNY_GEMM(4, 6, 8, 2)
    COND_LAUCNH_ONE_SKINNY_GEMM(4, 6, 16, 2)
    COND_LAUCNH_ONE_SKINNY_GEMM(4, 6, 16, 4)
    COND_LAUCNH_ONE_SKINNY_GEMM(4, 6, 32, 2)
    COND_LAUCNH_ONE_SKINNY_GEMM(4, 6, 32, 4)
    COND_LAUCNH_ONE_SKINNY_GEMM(5, 1, 8, 2)
    COND_LAUCNH_ONE_SKINNY_GEMM(5, 1, 8, 4)
    COND_LAUCNH_ONE_SKINNY_GEMM(5, 1, 8, 8)
    COND_LAUCNH_ONE_SKINNY_GEMM(5, 1, 16, 2)
    COND_LAUCNH_ONE_SKINNY_GEMM(5, 1, 16, 4)
    COND_LAUCNH_ONE_SKINNY_GEMM(5, 1, 16, 8)
    COND_LAUCNH_ONE_SKINNY_GEMM(5, 1, 32, 2)
    COND_LAUCNH_ONE_SKINNY_GEMM(5, 1, 32, 4)
    COND_LAUCNH_ONE_SKINNY_GEMM(5, 1, 32, 8)
    COND_LAUCNH_ONE_SKINNY_GEMM(5, 2, 8, 2)
    COND_LAUCNH_ONE_SKINNY_GEMM(5, 2, 8, 4)
    COND_LAUCNH_ONE_SKINNY_GEMM(5, 2, 16, 2)
    COND_LAUCNH_ONE_SKINNY_GEMM(5, 2, 16, 4)
    COND_LAUCNH_ONE_SKINNY_GEMM(5, 2, 16, 8)
    COND_LAUCNH_ONE_SKINNY_GEMM(5, 2, 32, 2)
    COND_LAUCNH_ONE_SKINNY_GEMM(5, 2, 32, 4)
    COND_LAUCNH_ONE_SKINNY_GEMM(5, 2, 32, 8)
    COND_LAUCNH_ONE_SKINNY_GEMM(5, 3, 8, 2)
    COND_LAUCNH_ONE_SKINNY_GEMM(5, 3, 16, 2)
    COND_LAUCNH_ONE_SKINNY_GEMM(5, 3, 16, 4)
    COND_LAUCNH_ONE_SKINNY_GEMM(5, 3, 32, 2)
    COND_LAUCNH_ONE_SKINNY_GEMM(5, 3, 32, 4)
    COND_LAUCNH_ONE_SKINNY_GEMM(5, 4, 8, 2)
    COND_LAUCNH_ONE_SKINNY_GEMM(5, 4, 16, 2)
    COND_LAUCNH_ONE_SKINNY_GEMM(5, 4, 16, 4)
    COND_LAUCNH_ONE_SKINNY_GEMM(5, 4, 32, 2)
    COND_LAUCNH_ONE_SKINNY_GEMM(5, 4, 32, 4)
    COND_LAUCNH_ONE_SKINNY_GEMM(5, 5, 8, 2)
    COND_LAUCNH_ONE_SKINNY_GEMM(5, 5, 16, 2)
    COND_LAUCNH_ONE_SKINNY_GEMM(5, 5, 16, 4)
    COND_LAUCNH_ONE_SKINNY_GEMM(5, 5, 32, 2)
    COND_LAUCNH_ONE_SKINNY_GEMM(5, 5, 32, 4)
    COND_LAUCNH_ONE_SKINNY_GEMM(5, 6, 16, 2)
    COND_LAUCNH_ONE_SKINNY_GEMM(5, 6, 32, 2)
    COND_LAUCNH_ONE_SKINNY_GEMM(6, 1, 8, 2)
    COND_LAUCNH_ONE_SKINNY_GEMM(6, 1, 8, 4)
    COND_LAUCNH_ONE_SKINNY_GEMM(6, 1, 8, 8)
    COND_LAUCNH_ONE_SKINNY_GEMM(6, 1, 16, 2)
    COND_LAUCNH_ONE_SKINNY_GEMM(6, 1, 16, 4)
    COND_LAUCNH_ONE_SKINNY_GEMM(6, 1, 16, 8)
    COND_LAUCNH_ONE_SKINNY_GEMM(6, 1, 32, 2)
    COND_LAUCNH_ONE_SKINNY_GEMM(6, 1, 32, 4)
    COND_LAUCNH_ONE_SKINNY_GEMM(6, 1, 32, 8)
    COND_LAUCNH_ONE_SKINNY_GEMM(6, 2, 8, 2)
    COND_LAUCNH_ONE_SKINNY_GEMM(6, 2, 8, 4)
    COND_LAUCNH_ONE_SKINNY_GEMM(6, 2, 16, 2)
    COND_LAUCNH_ONE_SKINNY_GEMM(6, 2, 16, 4)
    COND_LAUCNH_ONE_SKINNY_GEMM(6, 2, 16, 8)
    COND_LAUCNH_ONE_SKINNY_GEMM(6, 2, 32, 2)
    COND_LAUCNH_ONE_SKINNY_GEMM(6, 2, 32, 4)
    COND_LAUCNH_ONE_SKINNY_GEMM(6, 2, 32, 8)
    COND_LAUCNH_ONE_SKINNY_GEMM(6, 3, 8, 2)
    COND_LAUCNH_ONE_SKINNY_GEMM(6, 3, 16, 2)
    COND_LAUCNH_ONE_SKINNY_GEMM(6, 3, 16, 4)
    COND_LAUCNH_ONE_SKINNY_GEMM(6, 3, 32, 2)
    COND_LAUCNH_ONE_SKINNY_GEMM(6, 3, 32, 4)
    COND_LAUCNH_ONE_SKINNY_GEMM(6, 4, 8, 2)
    COND_LAUCNH_ONE_SKINNY_GEMM(6, 4, 16, 2)
    COND_LAUCNH_ONE_SKINNY_GEMM(6, 4, 16, 4)
    COND_LAUCNH_ONE_SKINNY_GEMM(6, 4, 32, 2)
    COND_LAUCNH_ONE_SKINNY_GEMM(6, 4, 32, 4)
    COND_LAUCNH_ONE_SKINNY_GEMM(6, 5, 16, 2)
    COND_LAUCNH_ONE_SKINNY_GEMM(6, 5, 32, 2)
    COND_LAUCNH_ONE_SKINNY_GEMM(6, 6, 16, 2)
    COND_LAUCNH_ONE_SKINNY_GEMM(6, 6, 32, 2)
    else { return 7;}

    return 0;
}


int skinny_gemm_fastpath(
    const fp8* __restrict__ A, const fp8* __restrict__ B, half* __restrict__ D, const float* scale_tensor,
    const int m, const int n, const int k
) {
    int b_stride = k;
    int split_k = SK_;

    int A_producers = A_PRODUCERS_;
    int B_producers = B_PRODUCERS_;
    int consumers = CONSUMERS_;

    int b_lanes = B_LANES_;
    int qsize = QSIZE_;
    int op_m = OP_M_;
    int ops = OPS_;

    hipStream_t stream = reinterpret_cast<hipStream_t>(0);

    return skinny_gemm(A, B, D, scale_tensor,
                m, n, k, b_stride, split_k,
                A_producers, B_producers, consumers,
                b_lanes, qsize, op_m, ops,
                stream);
}

int skinny_gemm_tb(
    torch::Tensor& A,
    torch::Tensor& B,
    torch::Tensor& D,
    torch::Tensor& scale_tensor,
    int64_t split_k,
    int64_t A_producers,
    int64_t B_producers,
    int64_t consumers,
    int64_t b_lanes,
    int64_t qsize,
    int64_t ops
) {
    // Retrieve pointers
    const fp8* __restrict__ A_ = (const fp8* __restrict__) A.data_ptr();
    const fp8* __restrict__ B_ = (const fp8* __restrict__) B.data_ptr();
    half* __restrict__ D_ = (half* __restrict__) D.data_ptr();
    float* __restrict__ scale_tensor_ = (float* __restrict__) scale_tensor.data_ptr();

    // Retrieve shapes
    const int m = A.size(0);
    const int n = B.size(1);
    const int k = A.size(1);
    const int b_stride = B.stride(1);

    // Depending on the number of rows in A, we have different OP_M
    int op_m;
    if (m <= 8)       { op_m = 8;  }
    else if (m <= 16) { op_m = 16; }
    else              { op_m = 32; }

    // Retrieve stream
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    // Device guard
    const at::cuda::OptionalCUDAGuard device_guard(device_of(A));

    // Launch kernel (branched on B_LANES)
    return skinny_gemm(
        A_, B_, D_, scale_tensor_,
        m, n, k, b_stride, split_k,
        A_producers, B_producers, consumers,
        b_lanes, qsize, op_m, ops,
        stream
    );
}


class HFRK_skinny_gemm {
private:
    int world_size;

public:
    HFRK_skinny_gemm(int world_size_)
        : world_size(world_size_) {
    }

    ~HFRK_skinny_gemm() {
    }

    int skinny_gemm_torch_binding(
        torch::Tensor& A,
        torch::Tensor& B,
        torch::Tensor& D,
        torch::Tensor& scale_tensor,
        int64_t split_k,
        int64_t A_producers,
        int64_t B_producers,
        int64_t consumers,
        int64_t b_lanes,
        int64_t qsize,
        int64_t ops
    ) {
        return skinny_gemm_tb(A, B, D, scale_tensor, split_k, A_producers, B_producers, consumers, b_lanes, qsize, ops);
    }
};


#define PYBIND11_MODULE_EXPAND(NAME, MODULE) PYBIND11_MODULE(NAME, MODULE)

PYBIND11_MODULE_EXPAND(TORCH_EXTENSION_NAME, m) {
    py::class_<HFRK_skinny_gemm>(m, "HFRK_skinny_gemm")
        .def(py::init<int>())
        .def("skinny_gemm_torch_binding", &HFRK_skinny_gemm::skinny_gemm_torch_binding);
}
