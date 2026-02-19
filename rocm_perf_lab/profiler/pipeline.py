import json
from typing import Optional
from rocm_perf_lab.hal.factory import build_arch_from_agent_metadata
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
    result = run_command(cmd, runs=runs, use_rocprof=use_rocprof, debug=debug)

    rocprof_data = result["rocprof"]

    if not rocprof_data or not rocprof_data.agent_metadata:
        raise RuntimeError("Agent metadata missing; profiling must be run with rocprof")

    arch = build_arch_from_agent_metadata(rocprof_data.agent_metadata)

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

            # Architecture-specific roofline handling
            if arch.arch_name == "gfx942":

                compute_metrics = [
                    "SQ_INSTS_VALU_FMA_F32",
                    "SQ_INSTS_VALU_ADD_F32",
                    "SQ_INSTS_VALU_MUL_F32",
                    "SQ_INSTS_VALU_MFMA_MOPS_F32",
                ]

                memory_metrics = [
                    "TCC_EA0_RDREQ",
                    "TCC_EA0_WRREQ",
                    "TCC_EA0_RDREQ_32B",
                    "TCC_EA0_WRREQ_32B",
                ]

                compute_values = run_with_rocprof_counters(cmd, compute_metrics, debug=debug)
                memory_values = run_with_rocprof_counters(cmd, memory_metrics, debug=debug)

                if compute_values and memory_values:
                    metrics = {**compute_values, **memory_values}

                    fma = metrics.get("SQ_INSTS_VALU_FMA_F32", 0.0)
                    add = metrics.get("SQ_INSTS_VALU_ADD_F32", 0.0)
                    mul = metrics.get("SQ_INSTS_VALU_MUL_F32", 0.0)
                    mfma_mops = metrics.get("SQ_INSTS_VALU_MFMA_MOPS_F32", 0.0)

                    # Scalar FLOPs
                    scalar_flops = 2.0 * fma + add + mul

                    # MFMA FLOPs (each MOP represents 512 flops)
                    mfma_flops = mfma_mops * 512.0

                    flops = scalar_flops + mfma_flops

                    rd = metrics.get("TCC_EA0_RDREQ", 0.0)
                    wr = metrics.get("TCC_EA0_WRREQ", 0.0)
                    rd32 = metrics.get("TCC_EA0_RDREQ_32B", 0.0)
                    wr32 = metrics.get("TCC_EA0_WRREQ_32B", 0.0)

                    # Assume remaining requests are 64B
                    rd64 = max(rd - rd32, 0.0)
                    wr64 = max(wr - wr32, 0.0)

                    bytes_moved = rd32 * 32.0 + rd64 * 64.0 + wr32 * 32.0 + wr64 * 64.0

                else:
                    flops = 0.0
                    bytes_moved = 0.0
            else:
                # Generic fallback: use raw VALU instruction count as FLOP proxy
                metrics = ["SQ_INSTS_VALU"]
                metric_values = run_with_rocprof_counters(cmd, metrics, debug=debug)

                if metric_values:
                    flops = metric_values.get("SQ_INSTS_VALU", 0.0)
                else:
                    flops = 0.0

                bytes_moved = 0.0

            runtime_s = result["mean_ms"] / 1000.0
            achieved_gflops = flops / runtime_s / 1e9 if runtime_s > 0 else 0.0
            achieved_bandwidth = bytes_moved / runtime_s / 1e9 if runtime_s > 0 else 0.0

            ai = flops / bytes_moved if bytes_moved > 0 else 0.0

            peak_compute = arch.peak_fp32_flops() / 1e9
            peak_bandwidth = memory_bandwidth_gbps or arch.theoretical_peak_bandwidth()

            bound = "compute"
            if peak_compute and peak_bandwidth and bytes_moved > 0:
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
            "architecture": arch.arch_name,
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

    peak = arch.peak_fp32_flops()
    profile_json["gpu"]["theoretical_peak_flops"] = peak

    # Validate schema
    try:
        from rocm_perf_lab.schema.profile import ProfileModel
        ProfileModel(**profile_json)
    except Exception as e:
        raise RuntimeError(f"Profile schema validation failed: {e}")

    return profile_json
