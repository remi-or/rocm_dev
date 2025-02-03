from typing import Optional, List, Dict
import argparse
import numpy as np
import json
import os
import subprocess
import itertools
import random

from project_class import Project

    
def skip(params: Dict[str, int]) -> bool:
    # Compute the amount of shared memory (smem) needed
    smem = params["QSIZE_"] * ((16 * params["B_LANES_"] + 8) * 64 * params["OPS"] + params["B_LANES_"] / 2)
    max_smem = 65536
    # Return True if we need to skip the benchmark of these parameters
    return any([
        params["QSIZE_"] < params["B_PRODUCERS_"] / params['B_LANES_'],                # queue is too short (B)
        (params["OPS"] == 1) and (params['QSIZE_'] % 2),                             # queue is odd but ops isn't
        (params["A_PRODUCERS_"] + params["B_PRODUCERS_"] + params["CONSUMERS_"]) > 16, # too many warps
        params["QSIZE_"] < max(params["A_PRODUCERS_"], params["CONSUMERS_"]),          # queue is too short (A/C)
        smem > max_smem,                                                            # not enough smem
    ])

def replace_line_in_buffer(buffer: List[str], prefix: str, replacement_line: str) -> None:
    for i, line in enumerate(buffer):
        if line.startswith(prefix):
            buffer[i] = replacement_line
            return None


class Grid:

    def __init__(self, 
        substitutions: Dict[str, List[int]],
        project: Project,
        arguments: str,
        shuffle: bool = False,
        skip_to: int = 0,
    ) -> None:
        self.project = project
        self.arguments = arguments
        self.params_enumerator = self.prepare_params_enumerator(substitutions, shuffle, skip_to)
        self.core = self.save_original_core()
        self.hash = (sum([sum(v) * len(k) for k, v in substitutions.items()]) % 10000) * 10000 + random.randint(0, 9999)
        self.log(f"{substitutions}\n\n")

    def prepare_params_enumerator(
        self, substitutions: Dict[str, List[int]], shuffle: bool, skip_to: int
    ) -> List[Dict[str, int]]:
        list_of_values = list(itertools.product(*substitutions.values()))
        if shuffle:
            random.seed(0)
            random.shuffle(list_of_values) 

        params_enumerator = []
        for i, values in enumerate(list_of_values):
            params = {k: v for k, v in zip(substitutions.keys(), values)}
            if skip(params):
                continue
            i += 1
            if i < skip_to:
                continue
            params_enumerator.append((i, params))
        
        return params_enumerator

    def save_original_core(self) -> List[str]:
        core_file = os.path.join(self.project.project_dir, "core.cu")
        with open(core_file) as file:
            original_core = file.readlines()
        return [core_file] + original_core

    def explore(self) -> None:
        best_latency, best_i, best_params = 1e9, -1, {}
        sep = "\n" + "-" * 80 + "\n"
        # Main loop
        for i, params in self.params_enumerator:

            # Bench current params
            data = self.bench_params(params)
            # Check if mean latency is the new best
            if data["mean"] < best_latency:
                best_i, best_latency, best_params = i, data["mean"], params
                self.log(f"{i} - {params}\n{data}{sep}NEW BEST: {best_i} at {best_latency}{sep}\n")
            else:
                self.log(f"{i} - {params}\n{data}\nBest is still: {best_i} at {best_latency}\n\n")

        # Rewrite best
        self.log(f"\n\nBEST OVERALL: {best_latency} at {best_i} with {best_params}")
        # Write back original core
        with open(self.core[0], "w") as file:
            for line in self.core[1:]:
                file.write(line)    

    def bench_params(self, params: Dict[str, int]) -> Dict[str, float]:
        # Create new core
        new_core = self.core[1:]
        for k, v in params.items():
            replace_line_in_buffer(new_core, f"#define {k}", f"#define {k} {v}\n")
        # Substitute old core with the new one
        with open(self.core[0], "w") as file:
            for line in new_core:
                file.write(line)
        # Bench with new core
        return self.bench_current_state(timeout=25)

    def log(self, msg: str) -> None:
        with open(f"tmp_{self.hash}.txt", "a") as file:
            file.write(msg)
        print(msg, end="")
        
    def bench_current_state(self, timeout: Optional[float] = None) -> Dict[str, float]:
        # Compile and run benchmark
        try:
            time_string = self.project.compile_and_run("bench.cu", self.arguments, timeout)
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
        # Process them
        data = {
            "mean": float(np.mean(bench_ts)),
            "std": float(np.std(bench_ts)),
            "median": float(np.median(bench_ts)),
        }
        return data



if __name__ == "__main__":

    substitutions = {
        "B_LANES_": [3, 4, 5],
        "OPS": [4],
        "A_PRODUCERS_": [1, 2, 3],
        "B_PRODUCERS_": [3, 4, 5, 6, 7, 8, 9],
        "CONSUMERS_": [1, 2, 3],
        "QSIZE_": [2, 3, 4],
        "SK": [2, 3, 4],
    }

    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", "-p", required=True)
    parser.add_argument("--arguments", "-a", default="")
    parser.add_argument("--shuffle", "-s", action="store_true")
    parser.add_argument("--restart", "-r", default=0, type=int)
    
    args = parser.parse_args()

    grid = Grid(
        substitutions=substitutions,
        project=Project(str(args.project)),
        arguments=str(args.arguments),
        shuffle=bool(args.shuffle),
        skip_to=int(args.restart),
    )

    grid.explore()
