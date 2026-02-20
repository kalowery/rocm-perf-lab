# ARCHITECTURE — rocm-perf-lab

This document describes the current system architecture as validated on gfx942-class GPUs (MI300X / MI325) using ROCm 7.1.

The system is structured as a deterministic profiling → analysis → guarded optimization pipeline.

---

# High-Level Flow

```
Application
   ↓
Base Profiling (rocprofv3 → .rocpd_profile)
   ↓
Roofline Analysis
   ↓
Trace-Derived Critical Path (DAG)
   ↓
ATT Deep Analysis (separate pass)
   ↓
Bottleneck Classification
   ↓
Guarded Closed-Loop Optimization
```

---

# 1. Profiling Layer

## 1.1 Base Profile (Authoritative Timing Source)

The base profiling pass produces a persistent `.rocpd_profile` database.

This database is the **only source of truth** for:

- Kernel dispatch timestamps
- Runtime accounting
- Critical path computation
- Roofline metrics

We explicitly avoid using ATT databases for critical-path or runtime computation.

Runtime is computed as:

```
SUM(dispatch_end - dispatch_start)
```

This ensures deterministic, trace-derived measurement.

## 1.2 ATT Profile (Extended Analysis)

ATT is executed in a separate profiling pass.

ATT-derived JSON provides:

- Wave state breakdown
- Stall reasons
- IPC estimates
- Occupancy information

ATT is used for feature enrichment only — never as the authoritative runtime source.

---

# 2. Roofline Layer

See `ROOFLINE_DESIGN.md` for implementation details.

Key properties:

- Architecture-aware FP32 FLOP scaling (CDNA3 VALU width = 8)
- MFMA integration
- Correct RDREQ / WRREQ byte accounting
- rocpd metric name → pmc_id mapping
- Deterministic runtime from dispatch timestamps

The roofline layer produces:

- FLOPs
- Bytes
- Operational intensity
- Achieved performance
- First-order bound classification

---

# 3. Trace-Derived DAG Engine

The DAG engine constructs a kernel execution graph from trace data.

Nodes:
- Kernel dispatches

Edges:
- Stream ordering
- Synchronization events
- Implicit ordering derived from timestamps

Critical path is computed using a longest-path algorithm over dispatch intervals.

The result is:

- `critical_path_ns`
- Per-kernel slack
- Criticality weight for optimization prioritization

This is a trace-derived execution DAG, not a purely logical dependency graph.

---

# 4. Bottleneck Classification

Features combine:

- Roofline position
- Achieved bandwidth vs peak
- Occupancy
- Stall breakdown
- Divergence indicators

Classifier is rule-based and deterministic.

Output:
- Single bottleneck label per kernel

---

# 5. Guarded HIP Optimizer (v1 Scope)

The optimizer operates under explicit constraints.

## 5.1 Scope

- Standalone HIP kernel source files (`.cu`)
- No whole-application refactoring
- No ABI or signature changes

## 5.2 Allowed Transformations (v1)

- Safe loop unrolling (factor 2–8)

Guardrails:

- No unrolling across `__syncthreads`
- No unrolling across atomics
- No unrolling across dynamic shared memory usage
- Respect wave64 execution model
- Avoid occupancy regressions (VGPR pressure awareness)

## 5.3 Static Guards

- Kernel signature invariance enforcement (textual/interface-level)
- Basic structural sanity checks (no claimed AST parsing)

## 5.4 Dynamic Guards

- Successful compilation via `hipcc`
- Deterministic rebuild
- Re-profile and re-measure
- Accept only if performance improves beyond threshold
- Automatic rollback on regression

---

# 6. Closed-Loop Control

Each optimization iteration:

1. Profile baseline (authoritative `.rocpd_profile`)
2. Select critical-path-weighted kernel
3. Generate transformation proposal (LLM)
4. Enforce signature invariants
5. Compile
6. If compilation fails → bounded repair loop (max 2 attempts)
7. Re-profile
8. Accept if improvement; otherwise reject

Loop terminates when:

- No improvement found
- Candidate pool exhausted
- Time budget exceeded

All decisions are logged in structured JSON.

---

# 7. Determinism and Safety Guarantees

- No binary patching
- No pointer rewriting
- No ABI changes
- No uncontrolled state mutation
- Regression automatically reverted
- Repair loop bounded

The system is empirical: only measured improvements are retained.

---

# 8. Validation Context

Validated on:

- AMD Instinct MI300X (gfx942)
- AMD Instinct MI325 (gfx942)
- ROCm 7.1
- rocprofv3

Closed-loop optimization validated end-to-end on real hardware.

---

Architecture is designed to parameterize device ceilings and vector widths for future AMD GPU generations.
