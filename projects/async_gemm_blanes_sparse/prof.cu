#include "./../common.cuh"
#include "./sparse_k.cu"

#include <hip/hip_fp16.h>

int main(int argc, char **argv) {
    HIP_CHECK( hipSetDevice(0) );

    const int m = 8;
    const int n = 6656;
    const int k = 16384;

    fp8 *dA, *dB;
    half* dD;
    float* dScale_tensor;

    empty_device_tensor<fp8>(dA, m * k);
    empty_device_tensor<fp8>(dB, n * k);
    zero_device_tensor<half>(dD, m * n);
    empty_device_tensor<float>(dScale_tensor, 1);

    async_gemm(dA, dB, dD, dScale_tensor, m, n, k);
}
