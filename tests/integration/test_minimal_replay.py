import os
import subprocess
from pathlib import Path


def run(cmd, env=None):
    subprocess.run(cmd, check=True, env=env)


def test_minimal_isolate_and_replay(tmp_path):
    repo_root = Path(__file__).resolve().parents[2]
    kernel_src = repo_root / "tests" / "integration" / "golden_kernel.cpp"

    # Build golden kernel
    binary = tmp_path / "golden_kernel"
    run(["hipcc", str(kernel_src), "-o", str(binary)])

    # Build paths to isolate shared library
    isolate_lib = repo_root / "rocm_perf_lab" / "isolate" / "tool" / "build" / "librocm_perf_isolate.so"
    assert isolate_lib.exists(), f"Isolate library not found at {isolate_lib}. Build isolate first."

    # Run kernel with isolate enabled
    env = os.environ.copy()
    env["HSA_TOOLS_LIB"] = str(isolate_lib)
    env["ISOLATE_KERNEL"] = "increment"
    env["ISOLATE_DISPATCH_INDEX"] = "0"

    run([str(binary)], env=env)

    captures = sorted(Path(".").glob("isolate_capture_*"))
    assert captures, "No isolate_capture_* directories found"
    capture_dir = captures[-1]

    assert (capture_dir / "dispatch.json").exists(), "dispatch.json missing"
    assert (capture_dir / "kernel.hsaco").exists(), "kernel.hsaco missing"

    # Run replay reserve-check
    run(["rocm-perf", "replay", "reserve-check", "--capture-dir", str(capture_dir)])

    # Run full VM replay
    run(["rocm-perf", "replay", "full-vm", "--capture-dir", str(capture_dir)])
