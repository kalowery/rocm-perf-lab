# USER GUIDE — rocm-perf-lab

rocm-perf-lab is a profiling and guarded optimization framework for HIP workloads targeting **gfx942-class GPUs (MI300X / MI325)**.

Validated on:

- AMD Instinct MI300X (gfx942)
- AMD Instinct MI325 (gfx942)
- ROCm 7.1
- rocprofv3

It combines architecture-aware roofline modeling, trace-derived critical path analysis, ATT deep inspection, and a safety-constrained closed-loop optimizer.

---

# 1. Profiling Workflow

## Baseline Profiling

```bash
rocm-perf-lab profile ./app
```

This performs:

- Base rocprofv3 profiling (authoritative `.rocpd_profile`)
- Hardware counter collection
- Roofline computation
- Critical path extraction
- Optional ATT deep analysis pass

Outputs include:

- `profile.json`
- `roofline.json`
- `dag.json`
- `analysis.json`

Runtime and critical path are derived from dispatch timestamps in the base profile.

---

# 2. Roofline Analysis

For each kernel, the system computes:

- FP32 FLOPs (CDNA3-aware scaling)
- DRAM bytes (RDREQ / WRREQ based)
- Operational intensity
- Achieved GFLOP/s
- Achieved bandwidth
- Memory vs compute bound regime

Roofline is first-order and feeds bottleneck classification.

See `ROOFLINE_DESIGN.md` for details.

---

# 3. Critical Path View

The DAG engine constructs a trace-derived execution graph from dispatch data.

It computes:

- `critical_path_ns`
- Per-kernel slack
- Optimization impact weighting

Optimizing non-critical kernels yields limited end-to-end speedup.

---

# 4. Bottleneck Categories

The classifier combines roofline + ATT-derived features.

Possible labels:

- Memory-bandwidth bound
- Latency bound
- Under-occupied
- Divergence limited
- Compute throughput limited

Classification is rule-based and deterministic.

---

# 5. Closed-Loop Optimization

```bash
rocm-perf-lab optimize ./app
```

## What It Does

1. Profile baseline (authoritative `.rocpd_profile`)
2. Rank kernels by critical-path impact
3. Generate guarded HIP transformation proposals
4. Enforce kernel signature invariants
5. Compile via `hipcc`
6. Re-profile
7. Accept only if measured performance improves

If compilation fails:

- A bounded repair loop (max 2 attempts) is attempted
- If still invalid → candidate is discarded

---

# 6. Optimizer Scope (v1)

The optimizer is intentionally constrained.

Scope:

- Standalone HIP kernel source files (`.cu`)
- No ABI changes
- No whole-application refactoring

Allowed transformation (v1):

- Safe loop unrolling (factor 2–8)

Guardrails:

- No unrolling across `__syncthreads`
- No unrolling across atomics
- No unrolling across dynamic shared memory usage
- Wave64-aware
- Avoid excessive VGPR pressure

This ensures safety and determinism.

---

# 7. Safety Model

- No binary patching
- No unsafe pointer rewriting
- Kernel signatures preserved
- Deterministic rebuild required
- Performance regression automatically reverted

Only empirically validated improvements are retained.

---

# 8. JSON Output Structure (Example)

Example `analysis.json` entry:

```json
{
  "kernel": "example_kernel",
  "time_ms": 2.41,
  "flops": 1048576,
  "bandwidth_gbps": 820.3,
  "operational_intensity": 0.125,
  "roofline_position": "memory_bound",
  "critical_path_score": 0.87,
  "bottleneck": "latency_bound"
}
```

Values are hardware-dependent.

---

For implementation details, see:

- `ARCHITECTURE.md`
- `DEVELOPER_GUIDE.md`
- `ROOFLINE_DESIGN.md`
