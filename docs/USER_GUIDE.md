# USER GUIDE

## Overview

rocm-perf-lab is a profiling and closed-loop optimization framework for HIP workloads on AMD MI300X (gfx942).

It combines roofline modeling, DAG-based critical path analysis, ATT deep inspection, and an LLM-driven guarded optimizer.

---

## Profiling Workflow

### Step 1: Baseline Profiling

```bash
rocm-perf-lab profile ./app
```

This performs:
- rocprofv3 counter collection (gfx942)
- Kernel timing capture
- ATT trace collection (optional)

Outputs:
- `profile.json`
- `roofline.json`
- `dag.json`
- `analysis.json`

---

## Roofline Analysis

For each kernel:

- Achieved FLOP/s
- Operational intensity
- Memory bandwidth
- Distance to compute roof

Helps answer:
- Is the kernel memory-bound?
- Is compute throughput saturated?

---

## Critical Path View

The DAG engine:
- Constructs kernel dependency graph
- Computes longest path
- Assigns impact score

Optimizing non-critical kernels yields minimal speedup.

---

## Bottleneck Categories

The classifier uses roofline + ATT features:

| Category | Symptoms |
|----------|----------|
| Memory-bound | High bandwidth, low intensity |
| Latency-bound | High stalls, low occupancy |
| Under-occupied | Low wave occupancy |
| Divergence-limited | High branch inefficiency |
| Compute-bound | Near compute roof |

---

## Closed-Loop Optimization

```bash
rocm-perf-lab optimize ./app
```

Process:
1. Profile
2. Rank critical kernels
3. Generate HIP rewrite candidates
4. Enforce signature invariants
5. Rebuild
6. Re-measure
7. Accept only if improved

If compilation fails:
- Compiler-repair loop attempts correction
- If still invalid → discard candidate

---

## Safety Guarantees

- No ABI changes
- Deterministic rebuild required
- Performance regression auto-reverted
- Optional numerical validation

---

## MI300X Notes

- Designed for gfx942 counters
- Uses rocprofv3 metric mapping
- Roofline ceilings derived from MI300X specs

---

## JSON Output Structure

`analysis.json` includes:

```json
{
  "kernel": "name",
  "time_ms": 1.23,
  "flops": 1.2e12,
  "bandwidth_gbps": 3500,
  "operational_intensity": 12.4,
  "roofline_position": "memory_bound",
  "critical_path_score": 0.87,
  "bottleneck": "latency_bound"
}
```

---

For deeper details, see ARCHITECTURE.md.
