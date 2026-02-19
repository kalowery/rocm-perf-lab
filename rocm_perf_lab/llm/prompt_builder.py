import re
from pathlib import Path
from typing import Dict, Any


def extract_dominant_kernel_code(source_text: str, kernel_name: str) -> str:
    kernel_base = kernel_name.split("(")[0]
    pattern = rf"__global__\s+void\s+{kernel_base}\s*\("
    match = re.search(pattern, source_text)
    if not match:
        return "// Dominant kernel not found in source."

    brace_start = source_text.find("{", match.end())
    if brace_start == -1:
        return "// Could not locate kernel body."

    brace_count = 0
    i = brace_start
    while i < len(source_text):
        if source_text[i] == "{":
            brace_count += 1
        elif source_text[i] == "}":
            brace_count -= 1
            if brace_count == 0:
                return source_text[match.start(): i + 1]
        i += 1

    return "// Failed to extract kernel body."


def build_optimization_context(
    source_path: Path,
    extended_profile: Dict[str, Any],
    full_source: bool = False,
) -> Dict[str, Any]:
    source_text = source_path.read_text()

    dominant_symbol = extended_profile.get("critical_path", {}).get("dominant_symbol")

    if full_source:
        code = source_text
    else:
        code = extract_dominant_kernel_code(source_text, dominant_symbol or "")

    context = {
        "hardware": extended_profile.get("gpu", {}),
        "kernel": extended_profile.get("kernel", {}),
        "runtime": {
            "runtime_ms": extended_profile.get("runtime_ms"),
            "stability": extended_profile.get("stability"),
        },
        "roofline": extended_profile.get("roofline", {}),
        "critical_path": extended_profile.get("critical_path", {}),
        "att": extended_profile.get("att", {}),
        "resources": extended_profile.get("resources", {}),
        "bottleneck": extended_profile.get("bottleneck", {}),
        "headroom_fraction": extended_profile.get("headroom_fraction"),
        "source": {
            "path": str(source_path),
            "kernel_name": dominant_symbol,
            "code": code,
        },
    }

    return context


def build_llm_prompt(context: Dict[str, Any], compact: bool = False) -> str:
    hw = context["hardware"]
    kernel = context["kernel"]
    roof = context["roofline"]
    cp = context["critical_path"]
    att = context["att"]
    bottleneck = context["bottleneck"]

    if compact:
        return (
            f"Optimize the following AMD GPU kernel (gfx942).\n"
            f"Runtime: {context['runtime']['runtime_ms']} ms\n"
            f"Bound: {roof.get('bound')}\n"
            f"Bottleneck: {bottleneck.get('primary')}\n"
            f"Stall fraction: {att.get('stall_fraction')}\n"
            f"Avg memory latency: {att.get('avg_memory_latency')} cycles\n"
            f"Headroom: {context.get('headroom_fraction')}\n"
            f"\nSource:\n{context['source']['code']}"
        )

    return f"""
You are optimizing a HIP kernel for AMD MI300X (gfx942).

=== Hardware ===
Architecture: {hw.get('architecture')}
Wave Size: {hw.get('wave_size')}
Compute Units: {hw.get('compute_units')}
Peak FLOPs: {hw.get('theoretical_peak_flops')}

=== Kernel ===
Name: {kernel.get('name')}
Grid: {kernel.get('grid')}
Block: {kernel.get('block')}
Runtime: {context['runtime']['runtime_ms']} ms

=== Roofline ===
Arithmetic Intensity: {roof.get('arithmetic_intensity')}
Achieved GFLOPs: {roof.get('achieved_gflops')}
Achieved Bandwidth (GB/s): {roof.get('achieved_bandwidth_gbps')}
Bound: {roof.get('bound')}

=== Critical Path ===
Dominant Kernel: {cp.get('dominant_symbol')}
Critical Path (ns): {cp.get('critical_path_ns')}
Fraction: {cp.get('fraction')}

=== ATT Analysis ===
Instruction Mix: {att.get('instruction_mix')}
Stall Fraction: {att.get('stall_fraction')}
Idle Fraction: {att.get('idle_fraction')}
Average Memory Latency: {att.get('avg_memory_latency')}
IPC: {att.get('ipc')}

=== Bottleneck ===
Primary: {bottleneck.get('primary')}
Confidence: {bottleneck.get('confidence')}
Reasoning: {bottleneck.get('reasoning')}

=== Headroom ===
Estimated Headroom Fraction: {context.get('headroom_fraction')}

=== Source Code ===
{context['source']['code']}

Provide:
1. Bottleneck explanation
2. Specific code-level optimizations
3. Expected performance impact
4. Trade-offs
"""
