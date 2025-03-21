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
    parser.add_argument("--test-runs", "-t", type=int, default=1)
    parser.add_argument("--skip-bench", "-sb", action="store_true")
    args = parser.parse_args()
    project = Project(str(args.project))
    message = str(args.message)
    arguments = str(args.arguments)
    test_runs = args.test_runs
    skip_bench = bool(args.skip_bench)

    # Compile and run test
    max_deltas = []
    sum_deltas = []
    for i in range(test_runs):
        # Run the test (after compiling if this is the first time)
        if i == 0:
            test_output = project.compile_and_run("test.cu", arguments)
        else:
            test_output = project.run("test.cu", arguments)
        test_output = test_output.replace("nan", "\"nan\"")
        # Parse the output if t is not 0
        try:
            parsed = json.loads(test_output)
            max_deltas.append(float(parsed["max_delta"]))
            sum_deltas.append(float(parsed["total_delta"]))
        except:
            print(test_output.replace("\\n", "\n"))
            raise Exception("Failed to parse the output")

    # Compile and run benchmark
    if skip_bench:
        times = ["End of warmup", 0]
    else:
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
        "max_deltas": f"{max(max_deltas):.4f} <- [" + ", ".join(map(lambda x: f"{x:.4f}", max_deltas)) + "]",
        "sum_deltas": f"{max(sum_deltas):.4f} <- [" + ", ".join(map(lambda x: f"{x:.4f}", sum_deltas)) + "]",
    }
    print(json.dumps(data, indent=4))

    # Backup the project
    hash_ = project.backup()
    project.record(hash_, arguments, data)
