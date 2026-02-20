---
name: rocm-perf-lab
description: Deterministic multi-kernel GPU performance analysis and guarded closed-loop optimization framework for ROCm 7.x using rocprofv3, rocpd, roofline modeling, ATT analysis, and critical-path–weighted kernel optimization.
---

# ROCm Perf Lab Skill

Use this skill when performing structured GPU performance engineering on ROCm-based AMD GPUs.

Assumptions:
- ROCm 7.x
- rocprofv3 available
- rocpd SQLite dispatch database available
- HIP kernels (standalone `.cu` sources for optimization)
- Applications may contain multiple kernels per execution

---

# Core Capabilities

## 1. Deterministic GPU Runtime Modeling

- Runtime derived from rocpd kernel dispatch timestamps
- Computed as SUM(dispatch_end - dispatch_start)
- Not based on host wall-clock time
- Works for multi-launch and multi-kernel workloads

The `.rocpd_profile` database is the authoritative timing source.

---

## 2. Multi-Kernel Critical Path Analysis

- Reconstructs dispatch DAG from rocpd trace data
- Supports cross-stream execution
- Computes:
  - `critical_path_ns`
  - Per-kernel slack
  - Critical-path contribution weighting

Optimization prioritization is based on measured critical-path impact, not isolated kernel time.

---

## 3. Architecture-Aware Roofline Modeling

Extracts and computes:

- FP32 FLOPs (CDNA3-aware VALU width scaling)
- MFMA contributions (if present)
- DRAM bytes using RDREQ / WRREQ counters
- Arithmetic intensity
- Achieved GFLOP/s
- Achieved GB/s
- First-order bound classification (memory vs compute)

Validated on gfx942-class GPUs (MI300X / MI325).

---

## 4. ATT Deep Analysis

Optional ATT pass provides:

- Wave occupancy
- Stall breakdown
- Instruction mix signals
- Latency indicators

ATT enriches feature extraction but does not define runtime.

---

## 5. Rule-Based Bottleneck Classification

Combines roofline position and ATT-derived features to produce deterministic labels such as:

- Memory-bandwidth bound
- Latency bound
- Under-occupied
- Divergence limited
- Compute throughput limited

Classification is deterministic for identical inputs.

---

## 6. Guarded Closed-Loop LLM Optimization

Command:

    rocm-perf-lab optimize <binary>

Scope (v1):
- Standalone HIP kernel source files (`.cu`)
- No ABI changes
- Kernel signature must remain identical
- Transformation limited to safe loop unrolling (factor 2–8)

Loop behavior:

1. Profile baseline (`.rocpd_profile` authoritative)
2. Rank kernels by critical-path contribution
3. Generate loop-unrolling proposal
4. Enforce signature invariance and basic structural checks
5. Compile via `hipcc`
6. Re-profile
7. Accept only if measured runtime improves
8. Automatically revert on regression

Compilation acts as the primary structural validator.

---

# What This Skill Does Not Do

- No AST-based structural verification
- No cross-file refactoring
- No automatic ABI changes
- No formal numerical equivalence proofs
- No speculative optimization without measurement

All improvements are empirical and hardware-validated.

---

# CLI Overview

Primary commands:

    rocm-perf-lab profile <binary>
    rocm-perf-lab optimize <binary>

See references/cli.md for additional details.
