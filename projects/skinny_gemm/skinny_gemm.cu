#include "./skinny_gemm_kernel.cu"

void skinny_gemm_notorch(
    const fp8* __restrict__ A,
    const fp8* __restrict__ B,
    half* __restrict__ D,
    const float* scale_tensor,
    const int m,
    const int n,
    const int k
) {
    // Depending on the number of rows in A, we have different OP_M
    int OP_M;
    if (m <= 8) {
        OP_M = 8;
    } else if (m <= 16) {
        OP_M = 16;
    } else {
        OP_M = 32;
    }

    // Deduce other constants
    const int OP_K = 512 / OP_M;
    const int WARPTILE_M = OP_M;
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
    dim3 grid(CU);
    dim3 block((A_PRODUCERS_ + B_PRODUCERS_ + CONSUMERS_) * WARPSIZE);

    // Dispatch to the correct kernel
    switch (OP_M) {
        // DEBUG: to avoid SMEM comparaison error
        // case 8:
        //     _skinny_gemm_kernel<B_LANES_, A_PRODUCERS_, B_PRODUCERS_, CONSUMERS_, QSIZE_, 8>
        //         <<<grid, block, 0, 0>>>
        //         (A, B, D, scale_tensor, m, n, k, k, SK);
        //     break;
        // case 16:
        //     _skinny_gemm_kernel<B_LANES_, A_PRODUCERS_, B_PRODUCERS_, CONSUMERS_, QSIZE_, 16>
        //         <<<grid, block, 0, 0>>>
        //         (A, B, D, scale_tensor, m, n, k, k, SK);
        //     break;
        case 32:
            _skinny_gemm_kernel<B_LANES_, QSIZE_, 32>
                <<<grid, block, 0, 0>>>
                (A, B, D, scale_tensor, A_PRODUCERS_, B_PRODUCERS_, CONSUMERS_, m, n, k, k, SK);
            break;
    }
}




// void skinny_gemm(
//     torch::Tensor& A,
//     torch::Tensor& B,
//     torch::Tensor& D,
//     torch::Tensor& scale_tensor,
//     int64_t b_lanes,
//     int64_t split_k
// ) {
//     // Compile-time constants
//     static constexpr int OP_M = 16;
//     static constexpr int OP_K = 32;
//     static constexpr int WARPTILE_M = OP_M;
//     static constexpr int WARPTILE_K = OP_K * OPS;

//     const int m = A.size(0);
//     const int n = B.size(1);
//     const int k = A.size(1);

//     const int B_stride = B.stride(1);

//     const fp8* __restrict__ A_ = (const fp8* __restrict__) A.data_ptr();
//     const fp8* __restrict__ B_ = (const fp8* __restrict__) B.data_ptr();
//     half* __restrict__ D_ = (half* __restrict__) D.data_ptr();
//     float* __restrict__ scale_tensor_ = (float* __restrict__) scale_tensor.data_ptr();

//     // Check shape
//     if (m > WARPTILE_M) {
//         std::cerr << "m = " << k << " is greater than WARPTILE_M = " << WARPTILE_M << std::endl;
//         exit(1);
//     }
//     if (k % WARPTILE_K != 0) {
//         std::cerr << "k = " << k << " is not divisible by WARPTILE_K = " << WARPTILE_K << std::endl;
//         exit(1);
//     }

//     // Prepare kernel launch
//     dim3 grid(CU, 1, 1);
//     dim3 block(1, 1, 1);
//     const at::cuda::OptionalCUDAGuard device_guard(device_of(A));
//     const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

//     // Launch kernel (branched on B_LANES)
//     switch (b_lanes) {
//         case 3:
//             block.x = WARPSIZE * (2 + 6 + 3);
//             _skinny_gemm_kernel<3, 2, 6, 3, 3><<<grid, block, 0, stream>>>(A_, B_, D_, scale_tensor_, m, n, k, B_stride, split_k);
//             break;
//         case 4:
//             block.x = WARPSIZE * (2 + 6 + 3);
//             _skinny_gemm_kernel<4, 2, 6, 3, 3><<<grid, block, 0, stream>>>(A_, B_, D_, scale_tensor_, m, n, k, B_stride, split_k);
//             break;
//         case 5:
//             block.x = WARPSIZE * (2 + 6 + 2);
//             _skinny_gemm_kernel<5, 2, 6, 2, 2><<<grid, block, 0, stream>>>(A_, B_, D_, scale_tensor_, m, n, k, B_stride, split_k);
//             break;
//         default:
//             break;
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
