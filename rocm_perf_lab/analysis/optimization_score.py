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
