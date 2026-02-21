# WHITEPAPER — rocm-perf-lab

## Abstract

rocm-perf-lab is a structured performance analysis and guarded optimization framework for HIP applications targeting **gfx942-class GPUs (MI300X / MI325)**. It combines hardware-counter–driven measurement, architecture-aware modeling, trace-derived execution analysis, and a safety-constrained optimization loop to produce *measured* performance improvements.

Unlike conventional profilers that stop at reporting metrics, rocm-perf-lab is designed to connect measurement to action — while preserving determinism, safety, and reproducibility.

Validated on ROCm 7.1 with rocprofv3.

---

# 1. Why This Exists

Modern GPUs expose thousands of performance counters. Tools such as rocprof can collect detailed traces, instruction counts, and memory statistics. However, raw metrics alone do not answer critical engineering questions:

- Which kernel actually limits end-to-end performance?
- Is a kernel limited by memory bandwidth, memory latency, or compute throughput?
- If we change the kernel, will overall runtime meaningfully improve?
- Can optimization be automated without breaking correctness or ABI contracts?

rocm-perf-lab exists to bridge the gap between *measurement* and *decision-making*.

It does so by combining three principles:

1. Measure only from authoritative hardware traces.
2. Interpret those traces using architecture-aware models.
3. Apply optimizations only when empirical improvement is verified.

---

# 2. Architecture-Aware Roofline Modeling

## 2.1 What Is a Roofline Model?

The roofline model is a first-order performance model that relates:

- **Arithmetic intensity** (FLOPs per byte moved)
- **Achieved compute throughput**
- **Achieved memory bandwidth**

It answers a fundamental question:

> Is this kernel limited by memory movement or by arithmetic throughput?

On GPUs, this distinction is crucial. Memory-bound kernels benefit from improving locality or reducing traffic. Compute-bound kernels benefit from increasing instruction-level efficiency.

## 2.2 Why Architecture Awareness Matters

Naively counting instructions leads to incorrect conclusions. On CDNA3 (gfx942), each VALU instruction operates on multiple FP32 lanes. rocm-perf-lab scales instruction counts using architecture-specific vector width (`fp32_valu_width = 8`) and incorporates MFMA contributions.

Similarly, DRAM traffic is derived from RDREQ/WRREQ counters with correct 32B/64B scaling rather than relying on aggregated byte counters.

Runtime is computed from dispatch timestamps in the base `.rocpd_profile` database:

```
SUM(dispatch_end - dispatch_start)
```

This ensures that all roofline calculations are grounded in measured hardware execution.

The result is a numerically consistent placement of kernels in memory-bound or compute-bound regimes.

---

# 3. Trace-Derived Critical Path Analysis

Optimizing the wrong kernel does not improve overall application runtime.

GPU applications frequently launch many kernels across streams. Even if a kernel is slow in isolation, it may not lie on the **critical path** — the chain of dependent execution that determines total runtime.

rocm-perf-lab constructs a kernel-level execution graph from dispatch timestamps and synchronization ordering. A longest-path algorithm determines:

- Total critical path duration
- Per-kernel slack
- Optimization priority weighting

This approach is trace-derived, not theoretical. It reflects how the application actually executed on hardware.

The practical consequence:

> Optimization effort is focused where it changes wall-clock time.

---

# 4. Microarchitectural Insight via ATT

Roofline classification tells us *what* kind of bottleneck exists. ATT (AMD Trace Tool) helps explain *why*.

A separate ATT profiling pass extracts:

- Wave occupancy
- Stall reason breakdown
- Issue utilization
- Latency signals

These features enable deeper diagnosis:

- High memory bandwidth but low occupancy → latency bound.
- High divergence → control-flow inefficiency.
- Low VALU utilization → under-occupied or instruction-bound.

ATT enriches analysis but does not define runtime. The base profile remains authoritative.

---

# 5. Deterministic Bottleneck Classification

Rather than presenting raw metrics, rocm-perf-lab produces a structured, rule-based bottleneck label per kernel.

Examples:

- Memory-bandwidth bound
- Latency bound
- Under-occupied
- Divergence limited
- Compute throughput limited

The classifier is deterministic. Given identical inputs, it produces identical outputs. This avoids ambiguity and makes optimization decisions reproducible.

---

# 6. Guarded Closed-Loop Optimization

## 6.1 The Problem With Naive Automation

Automatically rewriting GPU kernels is risky. Unconstrained transformations can:

- Break ABI contracts
- Change kernel signatures
- Introduce race conditions
- Reduce occupancy
- Degrade performance

rocm-perf-lab enforces strict guardrails.

## 6.2 Scope and Constraints (v1)

- Standalone HIP kernel source files (`.cu`)
- No ABI or signature changes
- No global state injection
- Limited to safe loop unrolling (factor 2–8)
- No transformations across synchronization or atomic regions

## 6.3 Empirical Acceptance

Each optimization iteration:

1. Profiles baseline (authoritative `.rocpd_profile`)
2. Generates a transformation proposal
3. Enforces static guards (signature invariance and basic structural checks; full AST validation is not currently implemented)
4. Compiles via `hipcc`
5. Performs bounded repair if needed (max 2 attempts)
6. Re-profiles
7. Accepts only if measured runtime improves

Regression is automatically reverted.

The optimizer is therefore *empirical*. Improvements are not assumed — they are measured.

---

# 7. Determinism and Safety Guarantees

The framework guarantees:

- Deterministic runtime accounting
- No binary patching
- No uncontrolled state mutation
- Bounded repair loops
- Automatic rollback on regression

The goal is not maximal automation, but safe, reproducible performance improvement.

---

# 8. Deterministic Kernel Isolation & VA-Faithful Replay

A key extension of rocm-perf-lab is the ability to perform **deterministic cross-process kernel reproduction**.

## 8.1 Motivation

Performance debugging and research often require isolating a single kernel dispatch and reproducing it outside the original application.

Traditional approaches rely on:

- Synthetic test harnesses
- Pointer rewriting
- Partial memory reconstruction

These approaches do not guarantee that the reproduced execution matches the original virtual memory layout or pointer topology.

rocm-perf-lab instead implements **strict virtual-address–faithful replay**.

---

## 8.2 Isolation Mechanism

The isolation tool uses the HSA Tools API (not LD_PRELOAD) to intercept queue creation and dispatch submission.

Captured artifacts:

```
isolate_capture/
    dispatch.json
    kernarg.bin
    kernel.hsaco
    memory_regions.json
    memory/region_<base>.bin
```

The snapshot includes:

- Original grid/workgroup dimensions
- Kernel object metadata
- Full device memory contents prior to dispatch
- Original virtual address layout

No pointer rewriting or relocation occurs during capture.

---

## 8.3 VM Reconstruction Model

Replay uses AMD virtual memory APIs:

- `hsa_amd_vmem_address_reserve`
- `hsa_amd_vmem_handle_create`
- `hsa_amd_vmem_map`
- `hsa_amd_vmem_set_access`

All reservations are page-aligned.

Replay aborts immediately if:

- Reservation fails
- Returned base ≠ requested aligned base
- Any mapping or copy step fails

Memory is reconstructed **before executable load** to avoid address conflicts.

This ensures strict pointer stability.

---

## 8.4 Reproducibility Guarantees

On CDNA-class GPUs (e.g., gfx942 / MI325), deterministic fixed-address reservation is supported.

Under these conditions, replay guarantees:

- Identical virtual addresses
- Identical kernel object
- Identical dispatch geometry
- Identical device memory state

The replayed kernel therefore executes in an environment equivalent to the original dispatch.

This provides a foundation for:

- Reproducible microarchitectural experiments
- Controlled performance regression testing
- Academic analysis of kernel behavior

---

# 9. Validation Context

Validated on:

- AMD Instinct MI300X (gfx942)
- AMD Instinct MI325 (gfx942)
- ROCm 7.1
- rocprofv3

Closed-loop optimization and VA-faithful replay have been exercised end-to-end on hardware.

---

# 10. Conclusion

rocm-perf-lab connects low-level hardware measurement to high-level optimization decisions. By combining architecture-aware modeling, trace-derived execution analysis, and strictly guarded program transformation, it enables performance improvements that are both measurable and safe.

The framework is designed for engineers who require numerical correctness, reproducibility, and disciplined optimization rather than heuristic or speculative automation.
