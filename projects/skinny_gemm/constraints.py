SIZEOF = {
    "fp8": 1,
    "int32": 4,
}

def fits_in_gemm(
    a_lanes: int,
    b_lanes: int,
    qsize: int,
    op_m: int,
    ops: int,
    max_smem: int = 65536,
    verbose: bool = False
) -> bool:

    # Infer other constants from op_m
    op_n = 32 if op_m == 32 else 16
    op_k = (16 * 32) / op_m
    if verbose:
        print(f"For {op_m = }, we got {op_n = } and {op_k = }")

    # Infer helper variables
    warptile_m = a_lanes * op_m
    warptile_n = b_lanes * op_n
    warptile_k = op_k * ops

    # Compute the amount of shared memory (smem) needed
    a_buffer = warptile_m * warptile_k * qsize * SIZEOF["fp8"]
    b_buffer = warptile_n * warptile_k * qsize * SIZEOF["fp8"]
    queue = 2 * b_lanes * qsize * SIZEOF["int32"] # * 4 is for int type
    smem = a_buffer + b_buffer + queue

    if verbose:
        fits = "fits" if smem <= max_smem else "does not fit"
        print(f"For {b_lanes = }, {qsize = }, {op_m = }, we got {smem = }: {fits}")

    return smem <= max_smem


if __name__ == "__main__":

    for a_lanes in [1, 2, 3, 4]:
        for b_lanes in [1, 2, 3, 4, 5, 6]:
            for qsize in [1, 2, 3, 4, 5, 6]:
                for op_m in [8, 16, 32]:
                    for ops in [4, 8]:

                        if op_m == 8 and a_lanes > 1:
                            continue

                        if fits_in_gemm(a_lanes, b_lanes, qsize, op_m, ops, verbose=False):
                            print(f"    COND_LAUCNH_ONE_SKINNY_GEMM({a_lanes}, {b_lanes}, {qsize}, {op_m}, {ops})")
