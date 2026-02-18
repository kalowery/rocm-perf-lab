import json
from typing import Optional
from rocm_perf_lab.hal.detect import detect_architecture
from .runner import run_command


def classify_cv(cv: float) -> str:
    if cv <= 0.05:
        return "stable"
    if cv <= 0.10:
        return "moderate"
    return "unstable"


def build_profile(
    cmd: str,
    runs: int = 3,
    use_rocprof: bool = True,
    clock_mhz: Optional[float] = None,
    debug: bool = False,
    roofline: bool = False,
    memory_bandwidth_gbps: Optional[float] = None,
):
    arch = detect_architecture()

    result = run_command(cmd, runs=runs, use_rocprof=use_rocprof, debug=debug)

    rocprof_data = result["rocprof"]

    resources = None
    occupancy = None

    if rocprof_data:
        resources = {
            "vgpr_per_thread": rocprof_data.vgpr_per_thread,
            "sgpr_per_wave": rocprof_data.sgpr_per_wave,
            "lds_bytes": rocprof_data.lds_bytes,
        }

        if rocprof_data.block and rocprof_data.vgpr_per_thread is not None:
            threads_per_block = (
                rocprof_data.block[0]
                * rocprof_data.block[1]
                * rocprof_data.block[2]
            )

            lds_bytes = rocprof_data.lds_bytes or 0

            occ = arch.compute_occupancy(
                vgpr_per_thread=rocprof_data.vgpr_per_thread,
                lds_per_block_bytes=lds_bytes,
                threads_per_block=threads_per_block,
            )

            occupancy = {
                "theoretical": occ,
                "threads_per_block": threads_per_block,
                "wave_size": arch.wave_size
            }

    roofline_data = None

    if roofline and use_rocprof:
        try:
            from rocm_perf_lab.profiler.rocprof_adapter import run_with_rocprof_counters

            metrics = ["SQ_INSTS_VALU", "TCC_READ_BYTES", "TCC_WRITE_BYTES"]
            metric_values = run_with_rocprof_counters(cmd, metrics, debug=debug)

            if metric_values:
                flops = 2.0 * metric_values.get("SQ_INSTS_VALU", 0.0)
                bytes_moved = (
                    metric_values.get("TCC_READ_BYTES", 0.0)
                    + metric_values.get("TCC_WRITE_BYTES", 0.0)
                )

                runtime_s = result["mean_ms"] / 1000.0
                achieved_gflops = flops / runtime_s / 1e9 if runtime_s > 0 else 0.0
                achieved_bandwidth = bytes_moved / runtime_s / 1e9 if runtime_s > 0 else 0.0

                ai = flops / bytes_moved if bytes_moved > 0 else 0.0

                peak_compute = arch.theoretical_peak_flops(clock_mhz) / 1e9 if clock_mhz else None
                peak_bandwidth = memory_bandwidth_gbps or arch.theoretical_peak_bandwidth()

                bound = "compute"
                if peak_compute and peak_bandwidth:
                    if peak_bandwidth * ai < peak_compute:
                        bound = "memory"

                roofline_data = {
                    "flops": flops,
                    "bytes": bytes_moved,
                    "arithmetic_intensity": ai,
                    "achieved_gflops": achieved_gflops,
                    "achieved_bandwidth_gbps": achieved_bandwidth,
                    "bound": bound,
                }
        except Exception:
            roofline_data = None

    profile_json = {
        "schema_version": "1.0",
        "kernel": {
            "name": rocprof_data.kernel_name if rocprof_data else None,
            "grid": rocprof_data.grid if rocprof_data else None,
            "block": rocprof_data.block if rocprof_data else None,
        },
        "gpu": {
            "architecture": arch.name,
            "wave_size": arch.wave_size,
            "compute_units": arch.compute_units,
        },
        "runtime_ms": result["mean_ms"],
        "stability": {
            "runs": result["runs"],
            "mean_ms": result["mean_ms"],
            "stddev_ms": result["stddev_ms"],
            "cv": result["cv"],
            "classification": classify_cv(result["cv"]),
        },
        "resources": resources,
        "occupancy": occupancy,
        "roofline": roofline_data,
    }

    if clock_mhz:
        peak = arch.theoretical_peak_flops(clock_mhz)
        profile_json["gpu"]["theoretical_peak_flops"] = peak

    # Validate schema
    try:
        from rocm_perf_lab.schema.profile import ProfileModel
        ProfileModel(**profile_json)
    except Exception as e:
        raise RuntimeError(f"Profile schema validation failed: {e}")

    return profile_json
