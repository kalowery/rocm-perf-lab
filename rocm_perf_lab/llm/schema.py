OPTIMIZATION_CONTEXT_SCHEMA = {
    "hardware": {
        "architecture": "string",
        "wave_size": "int",
        "compute_units": "int",
        "theoretical_peak_flops": "float",
    },
    "kernel": {
        "name": "string",
        "grid": "list[int]",
        "block": "list[int]",
    },
    "runtime": {
        "runtime_ms": "float",
        "stability": "object",
    },
    "roofline": {
        "flops": "float",
        "bytes": "float",
        "arithmetic_intensity": "float",
        "achieved_gflops": "float",
        "achieved_bandwidth_gbps": "float",
        "bound": "string",
    },
    "critical_path": {
        "critical_path_ns": "int",
        "dominant_symbol": "string",
        "fraction": "float",
    },
    "att": {
        "instruction_mix": "object",
        "stall_fraction": "float",
        "idle_fraction": "float",
        "avg_memory_latency": "float",
        "ipc": "float",
    },
    "resources": {
        "vgpr_per_thread": "int",
        "sgpr_per_wave": "int",
        "lds_bytes": "int",
        "occupancy": "object",
    },
    "bottleneck": {
        "primary": "string",
        "confidence": "float",
        "reasoning": "list[string]",
    },
    "headroom_fraction": "float",
    "source": {
        "path": "string",
        "kernel_name": "string",
        "code": "string",
    },
}
