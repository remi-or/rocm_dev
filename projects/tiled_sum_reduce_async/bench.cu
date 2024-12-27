#include "./../common.cuh"
#include "./src.cu"

int main(int argc, char **argv) {
    HIP_CHECK( hipSetDevice(0) );

    // Parameters
    const int iterations = 2500;
    const int warmups = 200;

    const int b = atoi(argv[1]);
    const int m = atoi(argv[2]);
    const int n = atoi(argv[3]);

    // Tensors
    float *hA, *dA, *hB, *dB, *D;
    random_host_tensor<float>(hA, b * m * n);
    random_host_tensor<float>(hB, b * m * n);
    // Events 
    hipEvent_t start, stop;
    HIP_CHECK(hipEventCreate(&start));
    HIP_CHECK(hipEventCreate(&stop));
    // Timer
    float t;
    float total_time = 0.0f;
    
    // Benchmarking loop
    for (int iter = 0; iter < (iterations + warmups); iter++) {

        // Create tensors
        tensor_h2d<float>(hA, dA, b * m * n);
        tensor_h2d<float>(hB, dB, b * m * n);
        empty_device_tensor<float>(D, m * n);
        // Flush cache 
        flush_device_cache();
        // Sync
        HIP_CHECK( hipDeviceSynchronize() );

        // Call and time kernel
        HIP_CHECK(hipEventRecord(start));
        tiled_sum_reduce(dA, dB, D, b, m, n);
        HIP_CHECK(hipEventRecord(stop));
        HIP_CHECK(hipEventSynchronize(stop));
        HIP_CHECK(hipEventElapsedTime(&t, start, stop));
        
        if (iter == warmups) {
            std::cout << "End of warmup, ";
        }
        std::cout << t * 1000 << ", ";

        // Free tensors
        HIP_CHECK(hipFree(dA));
        HIP_CHECK(hipFree(dB));
        HIP_CHECK(hipFree(D));
        // Sync
        HIP_CHECK( hipDeviceSynchronize() );
    }
}
