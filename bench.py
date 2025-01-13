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
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument("--skip-test", "-st", action="store_true")
    args = parser.parse_args()
    project = Project(str(args.project))
    message = str(args.message)
    arguments = str(args.arguments)
    verbose = bool(args.verbose)
    skip_test = bool(args.skip_test)

    # Compile and run test
    if skip_test:
        test_output = "SKIPPED"
    else:
        test_output = project.compile_and_run("test.cu", arguments)
    
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
    if verbose:
        print(json.dumps(data, indent=4))

    # Backup the project
    hash_ = project.backup()
    project.record(hash_, arguments, data)
