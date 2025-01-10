#pragma once

#include <hip/hip_runtime.h>
#include <iostream>

#define WARPSIZE 64

#define OP_PER_WARPTILE 2
#define OP_MN 16
#define OP_K 32

#define WARPTILE_M OP_MN
#define WARPTILE_N OP_MN
#define WARPTILE_K (OP_K * OP_PER_WARPTILE)
#define PRODUCED_MASK 257

#define PRODUCERS 7
#define CONSUMERS 2
#define QSIZE 28
#define G_ATOMICS true

#define ELEMS_PER_THREADS ((WARPTILE_M * WARPTILE_N) / WARPSIZE)
#define THREADS_PER_ROW (WARPTILE_N / ELEMS_PER_THREADS)
