#include "./skinny_gemm_kernel.cu"

#define COND_LAUCNH_ONE_SKINNY_GEMM(__bl, __qs, __om, __ops)                                         \
    else if (b_lanes == __bl && qsize == __qs && op_m == __om && ops == __ops) {                     \
        _skinny_gemm_kernel<__bl, __qs, __om, __ops><<<grid, block, 0, stream>>>(                    \
            A, B, D, scale_tensor, m, n, k, b_stride, split_k, A_producers, B_producers, consumers); \
    }

enum SkinnyGemmReturnCode {
    SUCCESS = 0,
    M_ABOVE_WARPTILE_M = 1,
    N_NOT_EVEN = 2,
    K_NOT_DIVISIBLE_BY_WARPTILE_K = 3,
    TOO_MANY_WARPS = 4,
    QSIZE_TOO_SMALL = 5,
    INVALID_CONFIG = 6
};

#ifdef SKINNY_GEMM_FULL_COMPILE
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
    // if (m > WARPTILE_M) {
    //     std::cerr << "m = " << m << " is greater than WARPTILE_M = " << WARPTILE_M << std::endl;
    //     return SkinnyGemmReturnCode::M_ABOVE_WARPTILE_M;
    // }
    if (n % 2 != 0) {
        std::cerr << "n = " << n << " is not even" << std::endl;
        return SkinnyGemmReturnCode::N_NOT_EVEN;
    }
    if (k % WARPTILE_K != 0) {
        std::cerr << "k = " << k << " is not divisible by WARPTILE_K = " << WARPTILE_K << std::endl;
        return SkinnyGemmReturnCode::K_NOT_DIVISIBLE_BY_WARPTILE_K;
    }

    // Check async
    if (A_producers + B_producers + consumers > 16) {
        std::cerr << "A_producers = " << A_producers << ", B_producers = " << B_producers << ", consumers = ";
        std::cerr << consumers << " is greater than 16" << std::endl;
        return SkinnyGemmReturnCode::TOO_MANY_WARPS;
    }
    if (qsize < A_producers || qsize < (B_producers / b_lanes) || qsize < consumers) {
        std::cerr << "qsize = " << qsize << " is less than A_producers = " << A_producers;
        std::cerr << ", B_producers / b_lanes = " << B_producers / b_lanes;
        std::cerr << ", or consumers = " << consumers << std::endl;
        return SkinnyGemmReturnCode::QSIZE_TOO_SMALL;
    }

    // Prepare kernel launch
    dim3 grid(CU);
    dim3 block((A_producers + B_producers + consumers) * WARPSIZE);

    // Dispatch to the correct kernel
    if (b_lanes == 0) {
        // This is a dummy if because the macro begins with an else if
        return SkinnyGemmReturnCode::INVALID_CONFIG;
    }
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
    else {
        return SkinnyGemmReturnCode::INVALID_CONFIG;
    }

    return SkinnyGemmReturnCode::SUCCESS;
}
#endif

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


    // Deduce other constants
    const int OP_K = 512 / op_m;
    const int WARPTILE_M = op_m;
    const int WARPTILE_K = OP_K * ops;

    // Check shapes
    if (n % 2 != 0) {
        std::cerr << "n = " << n << " is not even" << std::endl;
        return SkinnyGemmReturnCode::N_NOT_EVEN;
    }
    if (k % WARPTILE_K != 0) {
        std::cerr << "k = " << k << " is not divisible by WARPTILE_K = " << WARPTILE_K << std::endl;
        return SkinnyGemmReturnCode::K_NOT_DIVISIBLE_BY_WARPTILE_K;
    }

    // Check async
    if (A_producers + B_producers + consumers > 16) {
        std::cerr << "A_producers = " << A_producers << ", B_producers = " << B_producers << ", consumers = ";
        std::cerr << consumers << " is greater than 16" << std::endl;
        return SkinnyGemmReturnCode::TOO_MANY_WARPS;
    }
    if (qsize < A_producers || qsize < (B_producers / b_lanes) || qsize < consumers) {
        std::cerr << "qsize = " << qsize << " is less than A_producers = " << A_producers;
        std::cerr << ", B_producers / b_lanes = " << B_producers / b_lanes;
        std::cerr << ", or consumers = " << consumers << std::endl;
        return SkinnyGemmReturnCode::QSIZE_TOO_SMALL;
    }

    // Prepare kernel launch
    dim3 grid(CU);
    dim3 block((A_producers + B_producers + consumers) * WARPSIZE);

    // Dispatch to the correct kernel
    if (b_lanes == 0) {
        // This is a dummy if because the macro begins with an else if
        return SkinnyGemmReturnCode::INVALID_CONFIG;
    }
    COND_LAUCNH_ONE_SKINNY_GEMM(B_LANES_, QSIZE_, OP_M_, OPS_)
    else {
        return SkinnyGemmReturnCode::INVALID_CONFIG;
    }

    return SkinnyGemmReturnCode::SUCCESS;
}
