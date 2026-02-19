from dataclasses import dataclass


@dataclass
class OptimizationScore:
    headroom_fraction: float
    estimated_speedup: float
    priority_score: float


def compute_optimization_score(
    headroom_fraction: float,
    critical_path_fraction: float,
) -> OptimizationScore:
    """
    Compute conservative optimization score.

    priority_score determines whether optimization is worthwhile.
    """

    headroom_fraction = max(0.0, min(1.0, headroom_fraction))

    # Conservative speedup ceiling
    estimated_speedup = 1.0 / max(1e-6, (1.0 - headroom_fraction))

    priority_score = headroom_fraction * critical_path_fraction

    return OptimizationScore(
        headroom_fraction=headroom_fraction,
        estimated_speedup=estimated_speedup,
        priority_score=priority_score,
    )


# -------------------------------
# Regression Detection
# -------------------------------

from typing import Tuple, List, Dict

DEFAULT_REGRESSION_THRESHOLDS = {
    "runtime": 0.05,
    "vgpr": 0.20,
    "occupancy": 0.15,
    "lds": 0.30,
    "dram": 0.10,
}


def detect_regression(
    old: Dict,
    new: Dict,
    thresholds: Dict = DEFAULT_REGRESSION_THRESHOLDS,
) -> Tuple[bool, List[str]]:
    """
    Compare two extended profiling metric dicts.
    Returns (ok, reasons). ok=False means regression detected.
    """
    reasons: List[str] = []

    old_rt = old.get("runtime_ms")
    new_rt = new.get("runtime_ms")

    if old_rt and new_rt and new_rt > old_rt * (1 + thresholds["runtime"]):
        reasons.append("runtime_regression")

    def get_kernel_field(d: Dict, field: str):
        cp = d.get("critical_path", {})
        dom = cp.get("dominant_kernel_metrics", {})
        return dom.get(field)

    old_vgpr = get_kernel_field(old, "vgpr_per_wave")
    new_vgpr = get_kernel_field(new, "vgpr_per_wave")
    if old_vgpr and new_vgpr and new_vgpr > old_vgpr * (1 + thresholds["vgpr"]):
        reasons.append("vgpr_spike")

    old_occ = get_kernel_field(old, "occupancy")
    new_occ = get_kernel_field(new, "occupancy")
    if old_occ and new_occ and new_occ < old_occ * (1 - thresholds["occupancy"]):
        reasons.append("occupancy_drop")

    old_lds = get_kernel_field(old, "lds_bytes")
    new_lds = get_kernel_field(new, "lds_bytes")
    if old_lds and new_lds and new_lds > old_lds * (1 + thresholds["lds"]):
        reasons.append("lds_spike")

    old_dram = get_kernel_field(old, "dram_bytes")
    new_dram = get_kernel_field(new, "dram_bytes")
    if old_dram and new_dram and new_dram > old_dram * (1 + thresholds["dram"]):
        # Only flag if runtime did not improve meaningfully
        if not (old_rt and new_rt and new_rt < old_rt * 0.98):
            reasons.append("memory_traffic_increase")

    ok = len(reasons) == 0
    return ok, reasons
