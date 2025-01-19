#include "./../common.cuh"
#include "./src.cu"

#include <hip/hip_fp16.h>


int main(int argc, char **argv) {
    HIP_CHECK( hipSetDevice(0) );

    // Device tensors
    int* dbuffer;
    int* dout;
    zero_device_tensor<int>(dbuffer, 1);
    zero_device_tensor<int>(dout, 1);

    HIP_CHECK( hipDeviceSynchronize() );
    loop(dbuffer, dout);
    HIP_CHECK( hipDeviceSynchronize() );

    // Transfer result and free device tensors
    int* hout;
    tensor_d2h<int>(dout, hout, 1);
    HIP_CHECK( hipDeviceSynchronize() );
    HIP_CHECK(hipFree(dbuffer));
    HIP_CHECK(hipFree(dout));

    // Compare the two host tensors
    float delta;
    float sum_delta = 0.0f;
    float max_delta = 0.0f;
    for (int k = 0; k < 1; k++) {
        delta = abs((float) hout[k] - 29.0f);
        sum_delta += delta;
        max_delta = (delta > max_delta) ? delta : max_delta;
        std::cout << k << ":" << hout[k] << ", ";
    }
    std::cout << "{\"max_delta\": " << max_delta << ", \"total_delta\": " << sum_delta << "}";

    // Free host-side tensors
    free(hout);
}
