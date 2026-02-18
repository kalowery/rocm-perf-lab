import time
import subprocess
import statistics
from .rocprof_adapter import run_with_rocprof


def run_command(cmd: str, runs: int = 3, use_rocprof: bool = False, debug: bool = False):
    timings = []
    rocprof_data = None

    for _ in range(runs):
        if use_rocprof:
            result = run_with_rocprof(cmd, debug=debug)
            timings.append(result.kernel_time_ms)
            rocprof_data = result
        else:
            start = time.perf_counter()
            subprocess.run(cmd, shell=True, check=True)
            end = time.perf_counter()
            timings.append((end - start) * 1000)

    mean = statistics.mean(timings)
    stddev = statistics.stdev(timings) if len(timings) > 1 else 0.0
    cv = stddev / mean if mean > 0 else 0.0

    return {
        "mean_ms": mean,
        "stddev_ms": stddev,
        "cv": cv,
        "runs": runs,
        "rocprof": rocprof_data
    }
