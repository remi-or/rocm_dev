#include "./../common.cuh"
#include "./sparse_k.cu"

#define STACK 10
#define OUTD half

int main(int argc, char **argv) {
    HIP_CHECK( hipSetDevice(0) );

    // Parameters
    const int iterations = 3500 / STACK;
    const int warmups = 500 / STACK;

    assert(argc==4);
    const int m = atoi(argv[1]);
    const int n = atoi(argv[2]);
    const int k = atoi(argv[3]);

    // Tensors
    fp8 *hA; fp8* dA, *hB, *dB;
    random_host_tensor<fp8>(hA, m * k);
    random_host_tensor<fp8>(hB, k * n);
    OUTD *D;
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
        tensor_h2d<fp8>(hA, dA, m * k);
        tensor_h2d<fp8>(hB, dB, k * n);
        empty_device_tensor<OUTD>(D, m * n);
        // Flush cache 
        flush_device_cache();
        // Sync
        HIP_CHECK( hipDeviceSynchronize() );

        // Call and time kernel
        HIP_CHECK(hipEventRecord(start));
        #pragma unroll
        for (int i = 0; i < STACK; i++) {
            async_gemm(dA, dB, D, m, n, k);
        }
        HIP_CHECK(hipEventRecord(stop));
        HIP_CHECK(hipEventSynchronize(stop));
        HIP_CHECK(hipEventElapsedTime(&t, start, stop));
        
        if (iter == warmups) {
            std::cout << "End of warmup, ";
        }
        std::cout << t * 1000 / STACK << ", ";

        // Free tensors
        HIP_CHECK(hipFree(dA));
        HIP_CHECK(hipFree(dB));
        HIP_CHECK(hipFree(D));
        // Sync
        HIP_CHECK( hipDeviceSynchronize() );
    }
}
