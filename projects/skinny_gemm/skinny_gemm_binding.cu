// Torch-related
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/all.h>

#undef __HIP_NO_HALF_OPERATORS__
#undef __HIP_NO_HALF_CONVERSIONS__


// Main function
#include "./skinny_gemm_caller.cu"


// Torch bind of the main function
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
        : world_size(world_size_) { // TODO: remove that
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
