#include "./../common.cuh"
#include "./sparse_k.cu"

#define MUL 10
#define BATCH 5
#define OUTD half

int main(int argc, char **argv) {
    HIP_CHECK( hipSetDevice(7) );

    // Parameters
    const int warmups = (MUL * 100) / BATCH;
    const int iterations = (MUL * 250) / BATCH;

    assert(argc==4);
    const int m = atoi(argv[1]);
    const int n = atoi(argv[2]);
    const int k = atoi(argv[3]);

    // Tensors
    fp8 *hA, *dA, *hB, *dB;
    OUTD *D;
    float *dScale_tensor;

    random_host_tensor<fp8>(hA, m * k);
    random_host_tensor<fp8>(hB, BATCH * n * k);

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
        tensor_h2d<fp8>(hB, dB, BATCH * n * k);
        empty_device_tensor<OUTD>(D, m * n);
        random_device_tensor<float>(dScale_tensor, 1);
        // Flush cache 
        flush_device_cache();
        // Sync
        HIP_CHECK( hipDeviceSynchronize() );

        // Call and time kernel
        HIP_CHECK(hipEventRecord(start));
        #pragma unroll
        for (int i = 0; i < BATCH; i++) {
            skinny_gemm_notorch(
                dA, 
                dB + i * n * k, 
                D, 
                dScale_tensor, 
                m, n, k);
        }
        HIP_CHECK(hipEventRecord(stop));
        HIP_CHECK(hipEventSynchronize(stop));
        HIP_CHECK(hipEventElapsedTime(&t, start, stop));
        
        if (iter == warmups) {
            std::cout << "End of warmup, ";
        }
        std::cout << t * 1000 / BATCH << ", ";

        // Free tensors
        HIP_CHECK(hipFree(dA));
        HIP_CHECK(hipFree(dB));
        HIP_CHECK(hipFree(D));
        HIP_CHECK(hipFree(dScale_tensor));
    }
}


// PRE pstate to 4
//     "mean": 24.790065000000006,
//     "std": 0.04351972282770189,
//     "median": 24.78665,
//     "message": "",
//     "time_string": "",
//     "test_output": "SKIPPED"
// POST
//     "mean": 24.921865000000004,
//     "std": 0.033613914901421786,
//     "median": 24.9217,
//     "message": "",
//     "time_string": "",
//     "test_output": "SKIPPED"
