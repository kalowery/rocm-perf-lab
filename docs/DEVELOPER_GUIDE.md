# DEVELOPER GUIDE — rocm-perf-lab

This document describes the internal implementation details of rocm-perf-lab as validated on gfx942-class GPUs (MI300X / MI325) using ROCm 7.1.

The system is built around deterministic profiling, architecture-aware analysis, and a strictly guarded optimization loop.

---

# 1. Subsystem Overview

Core subsystems:

1. Profiling Engine (rocprofv3 + rocpd)
2. Roofline Analyzer (CDNA3-aware)
3. Trace-Derived Critical Path Engine
4. ATT Deep Analysis Module
5. Bottleneck Classifier (rule-based)
6. Guarded HIP Optimizer (v1 scope)
7. LLM Closed-Loop Controller

---

# 2. Profiling Engine

## 2.1 Base Profile

The base profiling pass generates a persistent `.rocpd_profile` database.

Authoritative data:

- Kernel dispatch start/end timestamps
- Hardware counter values
- Stream ordering

Metric name resolution is performed through the `rocpd_info_pmc` table (name → pmc_id).

The base profile is the only source of truth for:

- Runtime accounting
- Critical path computation
- Roofline metrics

## 2.2 Runtime Accounting

Runtime is computed as:

```
SUM(dispatch_end - dispatch_start)
```

This avoids ambiguity introduced by ATT databases.

## 2.3 ATT Profile (Separate Pass)

ATT runs in an independent profiling pass.

ATT-derived data provides:

- Wave occupancy
- Stall breakdown
- Issue utilization
- Memory latency signals

ATT enriches classification features but does not define runtime.

---

# 3. Roofline Module (CDNA3-Aware)

## 3.1 FLOP Estimation

```
FLOPs =
    SQ_INSTS_VALU * fp32_valu_width
  + MFMA_MOPS_F32 * 512
```

For gfx942 (CDNA3):

```
fp32_valu_width = 8
```

This corrects scalar undercounting assumptions.

## 3.2 DRAM Byte Accounting

Bytes are computed from RDREQ / WRREQ counters with 32B / 64B scaling.

Direct `TCC_*_BYTES` aggregation is intentionally avoided.

## 3.3 Bound Classification

First-order classification via peak bandwidth × AI comparison.

See `ROOFLINE_DESIGN.md` for full details.

---

# 4. Trace-Derived Critical Path Engine

Nodes:
- Kernel dispatches

Edges derived from:
- Stream ordering
- Synchronization events
- Timestamp ordering

Longest-path algorithm produces:

- `critical_path_ns`
- Slack per kernel
- Criticality weight

The DAG reflects measured execution, not static dependency inference.

---

# 5. Bottleneck Classifier

Deterministic rule-based classifier using:

- Roofline regime
- Achieved bandwidth vs peak
- Occupancy
- Stall fractions
- Divergence signals

Produces a single bottleneck label per kernel.

---

# 6. Guarded HIP Optimizer (v1 Scope)

## 6.1 Scope Constraints

- Standalone `.cu` kernel files only
- No ABI changes
- No signature modification
- No global state injection

## 6.2 Allowed Transformations (v1)

- Loop unrolling (factor 2–8)

Disallowed contexts:

- Across `__syncthreads`
- Across atomic operations
- Across dynamic shared memory usage

Architectural constraints:

- Wave64 execution model awareness
- Avoid excessive VGPR pressure
- Avoid occupancy collapse

## 6.3 Static Guards

- AST parsing
- Kernel signature invariance enforcement

## 6.4 Dynamic Guards

1. Compile via `hipcc`
2. Deterministic rebuild required
3. Re-profile using base profile pass
4. Accept only if performance improves beyond threshold
5. Automatic rollback otherwise

---

# 7. LLM Closed-Loop Controller

Iteration procedure:

1. Profile baseline (authoritative `.rocpd_profile`)
2. Select kernel weighted by critical path contribution
3. Generate transformation proposal
4. Enforce signature invariants
5. Compile
6. If compile fails → bounded repair loop (max 2 attempts)
7. Re-profile
8. Accept or reject based on measured performance

All iterations are logged in structured optimization trace JSON.

The loop is deterministic and bounded.

---

# 8. Extending the System

To add new analysis modules:

- Register metric extractor
- Extend JSON schema
- Update classifier feature mapping
- Maintain determinism in runtime accounting

For new architectures:

- Parameterize vector widths
- Update peak ceilings
- Validate counter mappings

---

# 9. Validation Context

Validated on:

- AMD Instinct MI300X (gfx942)
- AMD Instinct MI325 (gfx942)
- ROCm 7.1
- rocprofv3

Hardware validation required for performance-sensitive features.
