#include "./../common.cuh"
#include "./src.cu"


void host_tiled_sum_reduce(
    const float* __restrict__ A, 
    const float* __restrict__ B,
    float* __restrict__ &D, 
    const int m, 
    const int n, 
    const int k
) {
    float acc;
    const float* a;
    const float* b;

    // Allocate result tensor
    D = (float*) malloc(m * n * sizeof(float));

    // Square loop
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {

            // Setup K-wise loop
            acc = 0;
            a = A + i * k;
            b = B + j * k;
            // K-wise loop
            for (int l = 0; l < k; l++) {
                acc += a[l] * b[l];
            }
            // Store back
            D[i * n + j] = acc;
        }
    }
}

int main(int argc, char **argv) {
    HIP_CHECK( hipSetDevice(0) );

    assert(argc==4);
    const int m = atoi(argv[1]);
    const int n = atoi(argv[2]);
    const int k = atoi(argv[3]);

    // Host tensors
    float *hA, *hB, *host_ref;
    random_host_tensor<float>(hA, m * k);
    random_host_tensor<float>(hB, n * k);
    host_tiled_sum_reduce(hA, hB, host_ref, m, n, k);

    // Device tensors
    float *dA, *dB, *dD;
    tensor_h2d<float>(hA, dA, m * k);
    tensor_h2d<float>(hB, dB, n * k);
    zero_device_tensor<float>(dD, m * n);

    HIP_CHECK( hipDeviceSynchronize() );
    async_gemm(dA, dB, dD, m, n, k);
    HIP_CHECK( hipDeviceSynchronize() );

    // Transfer result and free device tensors
    float* host_result;
    tensor_d2h<float>(dD, host_result, m * n);
    HIP_CHECK( hipDeviceSynchronize() );
    HIP_CHECK(hipFree(dA));
    HIP_CHECK(hipFree(dB));
    HIP_CHECK(hipFree(dD));

    // Compare the two host tensors
    float delta;
    float sum_delta = 0.0f;
    float max_delta = abs(host_result[0] - host_ref[0]);
    for (int k = 0; k < m * n; k++) {
        delta = abs(host_result[k] - host_ref[k]);
        sum_delta += delta;
        max_delta = (delta > max_delta) ? delta : max_delta;
        std::cout << host_ref[k] << ":" << host_result[k] << ", ";
    }
    std::cout << "{\"max_delta\": " << max_delta << ", \"total_delta\": " << sum_delta << "}";

    // Free host-side tensors
    free(hA);
    free(hB);
    free(host_ref);
    free(host_result);
}
