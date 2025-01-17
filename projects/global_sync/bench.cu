#include "./src.cu"

int main(int argc, char **argv) {
    HIP_CHECK( hipSetDevice(0) );

    // Parameters
    const int iterations = 2500;
    const int warmups = 200;

    // Tensors
    uint* D;
    zero_device_tensor<uint>(D, 1024);
    // Events 
    hipEvent_t start, stop;
    HIP_CHECK(hipEventCreate(&start));
    HIP_CHECK(hipEventCreate(&stop));
    // Timer
    float t = 0;
    float total_time = 0.0f;
    std::cout << t * 1000 << ", ";
    
    // Benchmarking loop
    for (int iter = 0; iter < (iterations + warmups); iter++) {

        // Flush cache 
        flush_device_cache();
        // Sync
        HIP_CHECK( hipDeviceSynchronize() );

        // Call and time kernel
        HIP_CHECK(hipEventRecord(start));
        globs(D);
        HIP_CHECK(hipEventRecord(stop));
        HIP_CHECK(hipEventSynchronize(stop));
        HIP_CHECK(hipEventElapsedTime(&t, start, stop));
        
        if (iter == warmups) {
            std::cout << "End of warmup, ";
        }
        std::cout << t * 1000 << ", ";
    }
}
