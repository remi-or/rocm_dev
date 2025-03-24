#include "./skinny_gemm_kernel.cu"

#define LAUNCH_ONE_SKINNY_GEMM(B_LANES, QSIZE, OP_M) _skinny_gemm_kernel                      \
    <B_LANES_, QSIZE_, OP_M>                                                                  \
    <<<grid, block, 0, stream>>>                                                              \
    (A, B, D, scale_tensor, A_producers, B_producers, consumers, m, n, k, B_stride, split_k);

#define LAUNCH_SKINNY_GEMM(B_LANES, QSIZE)              \
    switch (OP_M) {                                     \
        case 8:                                         \
            LAUNCH_ONE_SKINNY_GEMM(B_LANES, QSIZE, 8);  \
            break;                                      \
        case 16:                                        \
            LAUNCH_ONE_SKINNY_GEMM(B_LANES, QSIZE, 16); \
            break;                                      \
        case 32:                                        \
            LAUNCH_ONE_SKINNY_GEMM(B_LANES, QSIZE, 32); \
            break;                                      \
    }


void skinny_gemm_notorch(
    const fp8* __restrict__ A,
    const fp8* __restrict__ B,
    half* __restrict__ D,
    const float* scale_tensor,
    const int m,
    const int n,
    const int k
) {

    // Deduce other constants
    const int OP_K = 512 / OP_M_;
    const int WARPTILE_M = OP_M_;
    const int WARPTILE_K = OP_K * OPS;

    // Check shape
    if (m > WARPTILE_M) {
        std::cerr << "m = " << m << " is greater than WARPTILE_M = " << WARPTILE_M << std::endl;
        exit(1);
    }
    if (n % 2 != 0) {
        std::cerr << "n = " << n << " is not even" << std::endl;
        exit(1);
    }
    if (k % WARPTILE_K != 0) {
        std::cerr << "k = " << k << " is not divisible by WARPTILE_K = " << WARPTILE_K << std::endl;
        exit(1);
    }

    // Prepare kernel launch
    int B_stride = k;
    int split_k = SK;
    int A_producers = A_PRODUCERS_;
    int B_producers = B_PRODUCERS_;
    int consumers = CONSUMERS_;
    hipStream_t stream = 0;

    dim3 grid(CU);
    dim3 block((A_producers + B_producers + consumers) * WARPSIZE);

    // Dispatch to the correct kernel
    if constexpr (OP_M_ == 8) {
        LAUNCH_ONE_SKINNY_GEMM(B_LANES_, QSIZE_, 8);
    }
    else if constexpr (OP_M_ == 16) {
        LAUNCH_ONE_SKINNY_GEMM(B_LANES_, QSIZE_, 16);
    }
    else {
        LAUNCH_ONE_SKINNY_GEMM(B_LANES_, QSIZE_, 32);
    }
}




// void skinny_gemm(
//     torch::Tensor& A,
//     torch::Tensor& B,
//     torch::Tensor& D,
//     torch::Tensor& scale_tensor,
//     int64_t A_producers,
//     int64_t B_producers,
//     int64_t consumers,
//     int64_t b_lanes,
//     int64_t split_k
// ) {
//     // Depending on the number of rows in A, we have different OP_M
//     int OP_M;
//     if (m <= 8)       { OP_M = 8;  }
//     else if (m <= 16) { OP_M = 16; }
//     else              { OP_M = 32; }

//     // Deduce other constants
//     const int OP_K = 512 / OP_M;
//     const int WARPTILE_M = OP_M;
//     const int WARPTILE_K = OP_K * OPS;

//     // Retrieve shapes
//     const int m = A.size(0);
//     const int n = B.size(1);
//     const int k = A.size(1);
//     const int B_stride = B.stride(1);

//     // Retrieve pointers
//     const fp8* __restrict__ A_ = (const fp8* __restrict__) A.data_ptr();
//     const fp8* __restrict__ B_ = (const fp8* __restrict__) B.data_ptr();
//     half* __restrict__ D_ = (half* __restrict__) D.data_ptr();
//     float* __restrict__ scale_tensor_ = (float* __restrict__) scale_tensor.data_ptr();

//     // Check shape
//     if (m > WARPTILE_M) {
//         std::cerr << "m = " << m << " is greater than WARPTILE_M = " << WARPTILE_M << std::endl;
//         exit(1);
//     }
//     if (n % 2 != 0) {
//         std::cerr << "n = " << n << " is not even" << std::endl;
//         exit(1);
//     }
//     if (k % WARPTILE_K != 0) {
//         std::cerr << "k = " << k << " is not divisible by WARPTILE_K = " << WARPTILE_K << std::endl;
//         exit(1);
//     }

//     // Prepare kernel launch
//     dim3 grid(CU);
//     dim3 block(WARPSIZE * (A_producers + B_producers + consumers));
//     const at::cuda::OptionalCUDAGuard device_guard(device_of(A));
//     const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

//     // Launch kernel (branched on B_LANES)
//     switch (b_lanes) {
//         case 2:
//             LAUNCH_SKINNY_GEMM(2, 4);
//         case 3:
//             LAUNCH_SKINNY_GEMM(3, 3);
//         case 4:
//             LAUNCH_SKINNY_GEMM(4, 3);
//         case 5:
//             LAUNCH_SKINNY_GEMM(5, 2);
//     }
// }


// class HFRK_skinny_gemm {
// private:
//     int world_size;

// public:
//     HFRK_skinny_gemm(int world_size_)
//         : world_size(world_size_) {
//     }

//     ~HFRK_skinny_gemm() {
//     }

//     void skinny_gemm16(
//         torch::Tensor& A,
//         torch::Tensor& B,
//         torch::Tensor& D,
//         torch::Tensor& scale_tensor,
//         int64_t b_lanes,
//         int64_t split_k
//     ) {
//         skinny_gemm(A, B, D, scale_tensor, b_lanes, split_k);
//     }
// };


// #define PYBIND11_MODULE_EXPAND(NAME, MODULE) PYBIND11_MODULE(NAME, MODULE)

// PYBIND11_MODULE_EXPAND(TORCH_EXTENSION_NAME, m) {
//     py::class_<HFRK_skinny_gemm>(m, "HFRK_skinny_gemm")
//         .def(py::init<int>())
//         .def("skinny_gemm16", &HFRK_skinny_gemm::skinny_gemm16);
// }
