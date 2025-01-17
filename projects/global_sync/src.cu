#include "./../common.cuh"
#include <iostream>
#include <hip/hip_cooperative_groups.h>

#define WARPSIZE 64

void __global__ globs_kernel(uint* D) {
    D[blockIdx.x] = 0;
    cooperative_groups::this_grid().sync();
    atomicDec(&D[blockIdx.x], 1);
}

void globs(uint* D) {
    dim3 grid(2, 1, 1);
    dim3 block(WARPSIZE, 1, 1);
    void* params[1];
    params[0] = &D;
    hipLaunchCooperativeKernel(
        globs_kernel,
        grid, 
        block,
        params,
        0, 0
    );
}
