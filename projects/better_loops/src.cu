#include "./../common.cuh"
#include <iostream>

#define WARPSIZE 64

void __global__ loop_kernel(int* buffer, int* out) {

    // Initialize shared queue
    volatile __shared__ int queue[1];
    if (threadIdx.x == 0) { 
        queue[0] = -1; 
    }
    __syncthreads();

    int warp_id = threadIdx.x / WARPSIZE;
    int q_state;
    int q_index = 0;

    // Producers
    if (warp_id == 0) {
        buffer[0] = 29;
        queue[0] = 1;
    } 
    
    // Consumers
    else {
        // while (queue[0] != 1) {
        //     asm volatile("s_sleep 0");
        // }
        asm volatile(
            "ds_read_b32 %0, %1 offset:0\n\t"
            "s_waitcnt lgkmcnt(0)\n\t"
            "v_cmp_ne_u32_e32 vcc, 1, %1\n\t"
            "s_cbranch_vccz -4"
            : "=v"(q_state)
            : "v"(q_index)
        );
        out[0] = buffer[0];
    }
}

void loop(int* buffer, int* out) {
    dim3 grid(1, 1, 1);
    dim3 block(2 * WARPSIZE, 1, 1);
    loop_kernel<<<grid, block, 0, 0>>>(buffer, out);
}


/*
Because smem is allocated as chunks of 512 bytes, if all shared variables can fit in that chunk, then the compiler does 
it that way. But if it's not the case, hten it breaks up stuff of different chunks of 512 bytes. We can avoid unexpected
behavior using only one big allocation, or try to bake in the pattern using defines and static offsets.
*/

//     // Consumers
//     else {
//         while (queue[0] != 1) {
//             asm volatile("s_sleep 0");
//         }
//         out[0] = buffer[0];
//     }
// }
