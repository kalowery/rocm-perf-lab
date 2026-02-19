import json
from dataclasses import dataclass
from pathlib import Path


@dataclass
class AttAnalysisResult:
    instruction_mix: dict
    stall_fraction: float
    idle_fraction: float
    avg_memory_latency: float
    ipc: float
    total_cycles: float


def _classify_instruction(isa: str) -> str:
    if isa.startswith("v_"):
        return "VALU"
    if isa.startswith("s_"):
        return "SALU"
    if isa.startswith("global_") or isa.startswith("flat_"):
        return "VMEM"
    if isa.startswith("ds_"):
        return "LDS"
    if "cbranch" in isa:
        return "Branch"
    if "mfma" in isa.lower():
        return "MFMA"
    return "Other"


def analyze_att(dispatch_dir: Path) -> AttAnalysisResult:
    code_json = dispatch_dir / "code.json"

    if not code_json.exists():
        raise RuntimeError(f"code.json not found in {dispatch_dir}")

    with open(code_json) as f:
        data = json.load(f)

    code_entries = data.get("code", [])

    total_hits = 0.0
    total_latency = 0.0
    total_stall = 0.0
    total_idle = 0.0

    class_hits = {}

    memory_latency_sum = 0.0
    memory_access_count = 0.0

    for entry in code_entries:
        isa = entry[0]
        hit = float(entry[6])
        latency = float(entry[7])
        stall = float(entry[8])
        idle = float(entry[9])

        total_hits += hit
        total_latency += latency
        total_stall += stall
        total_idle += idle

        cls = _classify_instruction(isa)
        class_hits[cls] = class_hits.get(cls, 0.0) + hit

        if cls == "VMEM":
            memory_latency_sum += latency
            memory_access_count += 1

    total_cycles = total_latency + total_stall + total_idle

    instruction_mix = {
        k: (v / total_hits if total_hits > 0 else 0.0)
        for k, v in class_hits.items()
    }

    stall_fraction = total_stall / total_cycles if total_cycles > 0 else 0.0
    idle_fraction = total_idle / total_cycles if total_cycles > 0 else 0.0
    ipc = total_hits / total_cycles if total_cycles > 0 else 0.0

    avg_memory_latency = (
        memory_latency_sum / memory_access_count
        if memory_access_count > 0
        else 0.0
    )

    return AttAnalysisResult(
        instruction_mix=instruction_mix,
        stall_fraction=stall_fraction,
        idle_fraction=idle_fraction,
        avg_memory_latency=avg_memory_latency,
        ipc=ipc,
        total_cycles=total_cycles,
    )
