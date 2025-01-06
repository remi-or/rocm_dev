#pragma once

#include <hip/hip_runtime.h>
#include <iostream>

#define WARPSIZE 64
#define WARPTILE_M 8
#define WARPTILE_N 32

#define PRODUCERS 6
#define CONSUMERS 3
#define QSIZE 12
#define PRODUCED_MASK 257

#define ELEMS_PER_THREADS ((WARPTILE_M * WARPTILE_N) / WARPSIZE)
#define THREADS_PER_ROW (WARPTILE_N / ELEMS_PER_THREADS)

using uint8 = unsigned char;
