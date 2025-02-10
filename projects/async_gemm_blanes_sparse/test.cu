#include "./../common.cuh"
#include "./skinny_gemm.cu"

#include <hip/hip_fp16.h>

template <typename out_dtype>
void host_skinny_gemm(
    const fp8* __restrict__ A, 
    const fp8* __restrict__ B,
    out_dtype* __restrict__ &D, 
    const float* scale_tensor, 
    const int m, 
    const int n, 
    const int k
) {
    float acc;
    const fp8* a;
    const fp8* b;

    // Allocate result tensor
    D = (out_dtype*) malloc(m * n * sizeof(out_dtype));

    // Square loop
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {

            // Setup K-wise loop
            acc = 0;
            a = A + i * k;
            b = B + j * k;
            // K-wise loop
            for (int l = 0; l < k; l++) {
                acc += __hip_cvt_fp8_to_halfraw(a[l], __HIP_E4M3_FNUZ).data * __hip_cvt_fp8_to_halfraw(b[l], __HIP_E4M3_FNUZ).data;
            }
            // Store back
            D[i * n + j] = (out_dtype) (acc * scale_tensor[0]);
        }
    }
}

#define OUTD half

int main(int argc, char **argv) {
    HIP_CHECK( hipSetDevice(7) );
    srand(0);

    assert(argc==4);
    const int m = atoi(argv[1]);
    const int n = atoi(argv[2]);
    const int k = atoi(argv[3]);

    // Host tensors
    fp8 *hA, *hB;
    OUTD* host_ref;
    float* hScale_tensor;

    fp8 elem = __hip_cvt_float_to_fp8(1.0f, __HIP_SATFINITE, __HIP_E4M3_FNUZ);
    random_host_tensor<fp8>(hA, m * k); //full_host_tensor<fp8>(hA, m * k, 0);
    random_host_tensor<fp8>(hB, n * k); //full_host_tensor<fp8>(hB, n * k, 0);

    // hA[(m-1) * k + 1184] = __hip_cvt_float_to_fp8(1, __HIP_SATFINITE, __HIP_E4M3_FNUZ);
    // for (int i = 0; i < k; i++) {
    //     hB[(n-1) * k + i] = __hip_cvt_float_to_fp8(i % 3, __HIP_SATFINITE, __HIP_E4M3_FNUZ);
    // }
    // hB[(n-1) * k + 1184] = __hip_cvt_float_to_fp8(1, __HIP_SATFINITE, __HIP_E4M3_FNUZ);
    // hB[(n-1) * k + 1952] = __hip_cvt_float_to_fp8(2, __HIP_SATFINITE, __HIP_E4M3_FNUZ);

    random_host_tensor<float>(hScale_tensor, 1);
    // hScale_tensor[0] = 1;
    host_skinny_gemm<OUTD>(hA, hB, host_ref, hScale_tensor, m, n, k);

    // Device tensors
    fp8 *dA, *dB;
    OUTD* dD;
    float* dScale_tensor;
    tensor_h2d<fp8>(hA, dA, m * k);
    tensor_h2d<fp8>(hB, dB, n * k);
    zero_device_tensor<OUTD>(dD, m * n);
    tensor_h2d<float>(hScale_tensor, dScale_tensor, 1);

    HIP_CHECK( hipDeviceSynchronize() );
    skinny_gemm_notorch(dA, dB, dD, dScale_tensor, m, n, k);
    HIP_CHECK( hipDeviceSynchronize() );

    // Transfer result and free device tensors
    OUTD* host_result;
    tensor_d2h<OUTD>(dD, host_result, m * n);
    HIP_CHECK( hipDeviceSynchronize() );
    HIP_CHECK(hipFree(dA));
    HIP_CHECK(hipFree(dB));
    HIP_CHECK(hipFree(dD));

    // Compare the two host tensors
    float delta;
    float sum_delta = 0.0f;
    float max_delta = 0.0f;
    for (int k = 0; k < m * n; k++) {
        delta = abs((float) host_result[k] - (float) host_ref[k]);
        sum_delta += delta;
        max_delta = (delta > max_delta) ? delta : max_delta;
        // std::cout << (float) host_ref[k] << ":" << (float) host_result[k] << ", ";
        // std::cout << k << ":" << (float) host_result[k] << ", ";
        // std::cout << k << ":" << delta << ", ";
    }
    std::cout << "{\"max_delta\": " << max_delta << ", \"total_delta\": " << sum_delta << "}";

    // Free host-side tensors
    free(hA);
    free(hB);
    free(host_ref);
    free(host_result);
}
