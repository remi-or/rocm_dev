#pragma once

// Includes
#include <hip/hip_runtime.h>
#include <hip/hip_fp8.h>


// Datatypes
using fp8 = __hip_fp8_storage_t;

using fp8_4 = int;

using fp8x8 = __attribute__( (__vector_size__(8 * sizeof(fp8)) )) fp8;
using fp8x16 = __attribute__( (__vector_size__(16 * sizeof(fp8)) )) fp8;

using fp8_4x2 = __attribute__( (__vector_size__(2 * sizeof(int)) )) int;
using fp8_4x4 = __attribute__( (__vector_size__(4 * sizeof(int)) )) int;

using f32x2 = __attribute__( (__vector_size__(2 * sizeof(float)) )) float;
using f32x4 = __attribute__( (__vector_size__(4 * sizeof(float)) )) float;

using uint8 = unsigned char;
using uint16 = unsigned short;
using uint32 = unsigned int;
using uint64 = unsigned long long;


// "CHECK" macros
#ifndef HIP_CHECK
#define HIP_CHECK(error)                    \
    if(error != hipSuccess)                       \
    {                                             \
        fprintf(stderr,                           \
                "Hip error: '%s'(%d) at %s:%d\n", \
                hipGetErrorString(error),         \
                error,                            \
                __FILE__,                         \
                __LINE__);                        \
        exit(EXIT_FAILURE);                       \
    }
#endif


// Data transfer
template <typename T>
void empty_device_tensor(T* &device_side, size_t size) {
    HIP_CHECK(hipMalloc(&device_side, size * sizeof(T)));
}

template <typename T>
void tensor_h2d(const T* host_tensor, T* &device_tensor, size_t size) {
    empty_device_tensor<T>(device_tensor, size);
    HIP_CHECK(hipMemcpy(device_tensor, host_tensor, size * sizeof(T), hipMemcpyHostToDevice));
}

template <typename T>
void tensor_d2h(const T* device_tensor, T* &host_tensor, size_t size) {
    host_tensor = (T*) malloc(size * sizeof(T));
    HIP_CHECK( hipMemcpy(host_tensor, device_tensor, size * sizeof(T), hipMemcpyDeviceToHost) );
}


// Data generation

template<typename T> 
void full_host_tensor(T* &x, size_t size, T elem) {
    x = (T*) malloc(size * sizeof(T));
    for (size_t i = 0; i < size; i++) {
        x[i] = elem;
    }
}

template<typename T> 
void zero_host_tensor(T* &x, size_t size) {
    full_host_tensor<T>(x, size, 0);
}

template <typename T>
void zero_device_tensor(T* &device_tensor, size_t size) {
    T* host_tensor;
    full_host_tensor<T>(host_tensor, size, 0);
    tensor_h2d<T>(host_tensor, device_tensor, size);
    free(host_tensor);
}

template<typename T> 
void random_host_tensor(T* &x, size_t size) {
    exit(1);
}
template<>
void random_host_tensor(float* &x, size_t size) {
    x = (float*) malloc(size * sizeof(float));
    for (size_t i = 0; i < size; i++) {
        x[i] = static_cast<float>(rand()) / static_cast<float>(RAND_MAX) - 0.5;
    }
}
template<>
void random_host_tensor(fp8* &x, size_t size) {
    x = (fp8*) malloc(size * sizeof(fp8));
    float y;
    for (size_t i = 0; i < size; i++) {
        y = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
        x[i] = __hip_cvt_float_to_fp8(2 * (y - 0.5f), __HIP_SATFINITE, __HIP_E4M3_FNUZ);
    }
}

template <typename T>
void random_device_tensor(T* &device_tensor, size_t size) {
    T* host_tensor;
    random_host_tensor<T>(host_tensor, size);
    tensor_h2d<T>(host_tensor, device_tensor, size);
    free(host_tensor);
}


// Cache flush
void __global__ _flush_cache_kernel(size_t l2_size, float* l2_sized_tensor) {
    // This kernel is always launched with 1024 threads and as many blocks as there are CUs

    // Flush shared memory
    static constexpr int smem_size = 16 * 1024;
    int smem_elems_per_threads = smem_size / (4 * 1024);
    const int smem_offset = threadIdx.x * smem_elems_per_threads;
    __shared__ float smem[smem_size / 4];
    #pragma unroll
    for (int i = 0; i < smem_elems_per_threads; i++) {
        ((float volatile *)smem)[smem_offset + i] = 0.0f;
    }

    // Flush L2
    int l2_elems_per_threads = l2_size / (4 * 1024);
    const int l2_offset = threadIdx.x * l2_elems_per_threads;
    l2_sized_tensor += l2_offset;
    float sum = 0.0f;
    for (int i = 0; i < l2_elems_per_threads; i++) {
        sum += l2_sized_tensor[i];
    }
    l2_sized_tensor[blockIdx.x] = sum;
}

void flush_device_cache() {
    int device_id;
    HIP_CHECK( hipGetDevice(&device_id) );
    hipDeviceProp_t prop;
    HIP_CHECK( hipGetDeviceProperties(&prop, device_id) );

    size_t l2_size = prop.l2CacheSize;
    float* l2_sized_tensor;
    empty_device_tensor<float>(l2_sized_tensor, l2_size);

    dim3 block(1024, 1, 1);
    unsigned int number_of_CUs = (unsigned int) prop.multiProcessorCount;
    dim3 grid(number_of_CUs, 1, 1);

    _flush_cache_kernel<<<grid, block, 0, 0>>>(l2_size, l2_sized_tensor);

    HIP_CHECK( hipFree(l2_sized_tensor) );
}
