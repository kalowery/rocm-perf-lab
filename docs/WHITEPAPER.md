# WHITEPAPER

## Abstract

rocm-perf-lab is a structured performance analysis and optimization framework for HIP applications targeting AMD Instinct MI300X (gfx942). It integrates hardware counter–based roofline modeling, critical-path DAG analysis, ATT-driven microarchitectural inspection, and a safety-constrained LLM optimization loop.

---

## Motivation

Traditional GPU profiling tools provide raw metrics but lack:
- End-to-end critical path prioritization
- Structured bottleneck classification
- Automated, safe optimization loops

rocm-perf-lab addresses these gaps.

---

## Methodology

### Roofline Modeling

Using rocprofv3 counters:

- Compute achieved FLOP/s
- Derive operational intensity
- Compare against MI300X ceilings

Enables first-order bottleneck detection.

---

### Critical Path Analysis

Kernel-level DAG constructed.

Longest-path algorithm identifies kernels whose optimization yields global speedup.

---

### ATT-Based Microarchitectural Insight

ATT traces provide per-wave stall breakdown.

This enables:
- Latency diagnosis
- Occupancy tuning
- Divergence analysis

---

### Bottleneck Classification

Combines roofline and ATT-derived features.

Produces deterministic kernel bottleneck labels.

---

### Closed-Loop LLM Optimization

LLM generates HIP kernel transformations.

Safety mechanisms:
- Signature invariants
- Compile validation
- Compiler-repair loop
- Performance gate
- Rollback guarantee

Only empirically validated improvements are retained.

---

## Safety and Determinism

The system ensures:

- No ABI changes
- No unsafe transformations
- Deterministic compilation required
- Regression auto-reverted

Optimization is empirical, not speculative.

---

## Validation

Evaluated on AMD Instinct MI300X (gfx942).

Demonstrated:
- Accurate roofline placement
- Reliable bottleneck classification
- Stable guarded optimization loop

---

## Conclusion

rocm-perf-lab combines structured GPU performance modeling with guarded AI-driven optimization to provide safe, architecture-aware performance improvement on MI300X.
