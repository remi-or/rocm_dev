from typing import Optional, List
import argparse
import numpy as np
import json
import os
import subprocess
import itertools
import random

from project_class import Project


def replace_line_in_buffer(buffer: List[str], prefix: str, replacement_line: str) -> None:
    for i, line in enumerate(buffer):
        if line.startswith(prefix):
            buffer[i] = replacement_line
            return None

def bench_current_state(
    project: Project,
    skip_test: bool,
    verbose: bool,
    timeout: Optional[float] = None,
) -> None:
    
    # Compile and run test
    if skip_test:
        test_output = "SKIPPED"
    else:
        test_output = project.compile_and_run("test.cu", arguments)
    
    # Compile and run benchmark
    try:
        time_string = project.compile_and_run("bench.cu", arguments, timeout)
        times = time_string[2:-3].split(", ")[3:]
    except subprocess.TimeoutExpired as t:
        print("ERROR: End of timeout.")
        return {"mean": 100000000000000000}
    except subprocess.CalledProcessError as t:
        print(f"ERROR: {t.stderr}.")
        return {"mean": 100000000000000000}
    
    # Retrieve benchmarked times
    bench_ts, t = [], times.pop()
    while t != "End of warmup":
        bench_ts.append(float(t))
        t = times.pop()
    warmup_ts = list(map(float, times))
    # Process them
    data = {
        "mean": float(np.mean(bench_ts)),
        "std": float(np.std(bench_ts)),
        "median": float(np.median(bench_ts)),
        # "test_output": test_output,
    }
    if verbose:
        print(json.dumps(data, indent=4))
    return data



if __name__ == "__main__":

    # Retrieve the name of the project
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", "-p", required=True)
    parser.add_argument("--arguments", "-a", default="")
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument("--restart", "-r", default=0, type=int)
    # parser.add_argument("--skip-test", "-st", action="store_true")
    args = parser.parse_args()
    project = Project(str(args.project))
    arguments = str(args.arguments)
    restart = int(args.restart)

    # Find core
    core_file = os.path.join(project.project_dir, "core.cu")
    with open(core_file) as file:
        original_core = file.readlines()

    # Rewrite core
    substitutions = {
        "B_LANES": [4, 5],
        "OPS": [2, 4],
        "A_PRODUCERS": [1, 2],
        "B_PRODUCERS": [8, 9, 10, 11, 12],
        "CONSUMERS": [1, 2, 3],
        "QSIZE": [2, 3, 4],
        "SK": [1, 2, 4, 8],
    }

    curr = -1
    best = 1000000000000000
    best_params = {}
    best_idx = -1

    hash = (sum([sum(v) * len(k) for k, v in substitutions.items()]) % 10000) * 1000 + random.randint(0, 999)
    log = open(f"tmp_{hash}.txt", "w")
    log.write(f"{substitutions = }\n\n")

    for values in itertools.product(*substitutions.values()):
        curr += 1
        params = {k: v for k, v in zip(substitutions.keys(), values)}

        # Skip cases
        skip = False
        if params["QSIZE"] < params["B_PRODUCERS"] / params['B_LANES']:
            skip = True
        if (params["OPS"] == 1) and (params['QSIZE'] % 2):
            skip = True
        if (params["A_PRODUCERS"] + params["B_PRODUCERS"] + params["CONSUMERS"]) > 16:
            skip = True
        if params["QSIZE"] < max(params["A_PRODUCERS"], params["CONSUMERS"]):
            skip = True

        smem = params["QSIZE"] * ((16 * params["B_LANES"] + 8) * 64 * params["OPS"] + params["B_LANES"] / 2)
        if smem > 65536:
            skip = True

        skip_to = restart
        if curr < skip_to:
            skip = True

        # Maybe skip
        msg = f"{curr} - {params}: " + ("SKIPPED" if skip else "") + f"\t\t(best = {best:.2f} at {best_idx})"
        log.write(msg + "\n")
        print(msg)
        if skip:
            continue

        # Create new core buffer
        current_core = original_core + []
        # Alter it
        for k, v in params.items():
            replace_line_in_buffer(current_core, f"#define {k}", f"#define {k} {v}\n")
        # Write it back
        with open(core_file, "w") as file:
            for line in current_core:
                file.write(line)

        # Bench with altered core
        data = bench_current_state(
            project=project, 
            skip_test=True, 
            verbose=True,
            timeout=15,
        )
        print()
        log.write(f"{data}\n")
        log.close()
        log = open(f"tmp_{hash}.txt", "a")

        # Memoize best
        if data["mean"] and data["mean"] < best:
            best = data["mean"]
            best_idx = curr
            best_params = params
            print("-" * 80, f"NEW BEST: {best}", "-" * 80, sep="\n")

    # Output best 
    log.write(f"{best = }\n{best_params = }")
    print(f"{best = }\n{best_params = }")
        
    # Write back original core
    with open(core_file, "w") as file:
        for line in original_core:
            file.write(line)        
