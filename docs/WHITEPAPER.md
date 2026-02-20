# WHITEPAPER — rocm-perf-lab

## Abstract

rocm-perf-lab is a structured performance analysis and guarded optimization framework for HIP applications targeting **gfx942-class GPUs (MI300X / MI325)**.

Validated on ROCm 7.1 with rocprofv3, the system integrates:

- Architecture-aware roofline modeling
- Trace-derived critical-path analysis
- ATT-based microarchitectural inspection
- A safety-constrained, empirically validated LLM optimization loop

The design emphasizes determinism, numerical correctness, and rollback-safe optimization.

---

## Motivation

Traditional GPU profilers expose raw metrics but do not:

- Quantify end-to-end critical path impact
- Provide structured bottleneck classification
- Offer empirically guarded optimization loops

rocm-perf-lab addresses these gaps by combining hardware-derived measurements with constrained program transformations.

---

## Methodology

### 1. Architecture-Aware Roofline Modeling

Using rocprofv3 counters resolved via rocpd, the system computes:

- FP32 FLOPs with CDNA3-aware VALU scaling
- MFMA contributions
- DRAM traffic from RDREQ / WRREQ counters
- Operational intensity and achieved throughput

Runtime is derived from kernel dispatch timestamps in the base `.rocpd_profile` database.

This avoids ambiguity introduced by auxiliary trace databases.

Roofline classification is first-order and feeds bottleneck analysis.

---

### 2. Trace-Derived Critical Path Analysis

A kernel-level execution DAG is constructed from dispatch timestamps and synchronization ordering.

The longest-path algorithm produces:

- `critical_path_ns`
- Per-kernel slack
- Optimization priority weights

The graph reflects measured execution behavior rather than static dependency inference.

---

### 3. ATT-Based Microarchitectural Analysis

A separate ATT profiling pass extracts:

- Wave occupancy
- Stall reason breakdown
- Issue utilization
- Memory latency signals

These features enrich bottleneck classification but do not define runtime.

---

### 4. Deterministic Bottleneck Classification

A rule-based classifier combines:

- Roofline regime
- Achieved bandwidth vs peak
- Occupancy metrics
- Stall fractions
- Divergence indicators

The classifier produces deterministic labels (e.g., memory-bound, latency-bound).

---

### 5. Guarded Closed-Loop Optimization

The optimization loop operates under strict constraints:

- No ABI changes
- Kernel signature invariance enforced
- Standalone `.cu` kernel scope (v1)
- Limited to safe loop unrolling (factor 2–8)
- No transformations across synchronization or atomic regions

Each iteration:

1. Profiles baseline (authoritative `.rocpd_profile`)
2. Generates a transformation proposal
3. Enforces static guards
4. Compiles via `hipcc`
5. Performs bounded repair if needed (max 2 attempts)
6. Re-profiles
7. Accepts only if measured performance improves

Regression is automatically reverted.

Optimization is empirical, not speculative.

---

## Safety and Determinism

The system guarantees:

- Deterministic runtime accounting
- No uncontrolled state mutation
- No binary patching
- Bounded repair loops
- Automatic rollback on regression

All improvements are validated against measured hardware execution.

---

## Validation

Validated on:

- AMD Instinct MI300X (gfx942)
- AMD Instinct MI325 (gfx942)
- ROCm 7.1
- rocprofv3

Closed-loop optimization has been exercised end-to-end on real hardware with regression detection enabled.

---

## Conclusion

rocm-perf-lab integrates architecture-aware GPU performance modeling with a strictly guarded, empirically validated optimization loop.

The framework prioritizes numerical correctness, determinism, and safety over unconstrained automation, making it suitable for performance-sensitive HIP workloads on gfx942-class GPUs.
