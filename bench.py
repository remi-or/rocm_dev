import argparse
import numpy as np
import json

from project_class import Project



if __name__ == "__main__":

    # Retrieve the name of the project
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", "-p", required=True)
    parser.add_argument("--message", "-m", default="")
    parser.add_argument("--arguments", "-a", default="")
    args = parser.parse_args()
    project = Project(str(args.project))
    message = str(args.message)
    arguments = str(args.arguments)

    # Compile and run test
    test_output = project.compile_and_run("test.cu", arguments)
    print(f"{test_output = }")
    
    # Compile and run benchmark
    time_string = project.compile_and_run("bench.cu", arguments)
    times = time_string[2:-3].split(", ")[3:]
    
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
        "message": message,
        "time_string": "", # str(time_string),
        "test_output": test_output,
    }

    # Backup the project
    hash_ = project.backup()
    project.record(hash_, arguments, data)
