#include "./../common.cuh"
#include <iostream>

#define WARPSIZE 64
#define CONSUMERS 2

void __global__ barrier_kernel(int* buffer, int* out) {

    // Initialize shared queue
    __shared__ int queue[1];
    if (threadIdx.x == 0) {
        queue[threadIdx.x] = 0;
    }
    __syncthreads();

    int warp_id = threadIdx.x / WARPSIZE;

    // Producers
    if (warp_id < CONSUMERS) {
        buffer[0] = 29;
        queue[warp_id] = 1;
        // asm volatile("s_wakeup");
        // asm volatile("s_wakeup");
        // __syncthreads();
    } 
    
    // Consumers
    else {
        if (queue[0] != 1) {
            asm volatile("s_sleep 100");
        }
        out[0] = buffer[0];
    }
}

void barrier(int* buffer, int* out) {
    dim3 grid(1, 1, 1);
    dim3 block(2 * WARPSIZE, 1, 1);
    barrier_kernel<<<grid, block, 0, 0>>>(buffer, out);
}
