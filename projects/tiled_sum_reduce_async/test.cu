#include "./../common.cuh"
#include "./src.cu"


void host_tiled_sum_reduce(
    const float* __restrict__ A, 
    const float* __restrict__ B,
    float* __restrict__ &D, 
    const int b, 
    const int m, 
    const int n
) {
    // Allocate result tensor
    D = (float*) malloc(m * n * sizeof(float));
    for (int k = 0; k < m * n; k++) {
        D[k] = 0.0f;
    }
    // Batch-wise loop
    for (int ib = 0; ib < b; ib++) {
        for (int k = 0; k < m * n; k++) {
            D[k] += A[k] + B[k];
        }
        A += m * n;
        B += m * n;
    }
}

int main(int argc, char **argv) {
    HIP_CHECK( hipSetDevice(0) );

    assert(argc==4);
    const int b = atoi(argv[1]);
    const int m = atoi(argv[2]);
    const int n = atoi(argv[3]);

    // Host tensors
    float *hA, *hB, *host_ref;
    random_host_tensor<float>(hA, b * m * n);
    random_host_tensor<float>(hB, b * m * n);
    host_tiled_sum_reduce(hA, hB, host_ref, b, m, n);

    // Device tensors
    float *dA, *dB, *dD;
    tensor_h2d<float>(hA, dA, b * m * n);
    tensor_h2d<float>(hB, dB, b * m * n);
    zero_device_tensor<float>(dD, m * n);

    HIP_CHECK( hipDeviceSynchronize() );
    tiled_sum_reduce(dA, dB, dD, b, m, n);
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
        // std::cout << host_ref[k] << ":" << host_result[k] << ", ";
    }
    std::cout << "{\"max_delta\": " << max_delta << ", \"total_delta\": " << sum_delta << "}";

    // Free host-side tensors
    free(hA);
    free(hB);
    free(host_ref);
    free(host_result);
}
