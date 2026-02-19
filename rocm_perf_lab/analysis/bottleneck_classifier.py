from dataclasses import dataclass
from typing import Optional


@dataclass
class BottleneckResult:
    primary: str
    secondary: Optional[str]
    confidence: float
    reasoning: list


def classify_bottleneck(
    stall_fraction: float,
    instruction_mix: dict,
    roofline_bound: Optional[str],
    avg_memory_latency: float,
) -> BottleneckResult:

    reasoning = []
    primary = "Unknown"
    secondary = None
    confidence = 0.5

    valu = instruction_mix.get("VALU", 0.0)
    salu = instruction_mix.get("SALU", 0.0)
    vmem = instruction_mix.get("VMEM", 0.0)
    branch = instruction_mix.get("Branch", 0.0)

    # Memory latency bound
    if stall_fraction > 0.30 and avg_memory_latency > 200 and vmem < 0.20:
        primary = "Memory Latency Bound"
        confidence = 0.8
        reasoning.append("High stall fraction with high memory latency.")

    # Memory bandwidth bound
    elif roofline_bound == "memory" and vmem > 0.20:
        primary = "Memory Bandwidth Bound"
        confidence = 0.75
        reasoning.append("Roofline indicates memory-bound behavior.")

    # Compute bound
    elif roofline_bound == "compute" and valu > 0.60 and stall_fraction < 0.20:
        primary = "Compute Bound"
        confidence = 0.75
        reasoning.append("High VALU fraction with low stall.")

    # Scalar bottleneck
    elif salu > 0.25:
        primary = "Scalar Bottleneck"
        confidence = 0.7
        reasoning.append("High SALU instruction fraction.")

    # Control flow
    elif branch > 0.15:
        primary = "Control Flow Divergence"
        confidence = 0.65
        reasoning.append("High branch fraction.")

    else:
        primary = "Mixed / Unclear"
        confidence = 0.5
        reasoning.append("No dominant bottleneck detected.")

    return BottleneckResult(
        primary=primary,
        secondary=secondary,
        confidence=confidence,
        reasoning=reasoning,
    )
