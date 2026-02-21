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

# 8. Kernel Isolation & VA-Faithful Replay

The system now includes a **kernel isolation and full virtual-address (VA) faithful replay subsystem**.

## 8.1 Isolation Tool (HSA Tools API)

The isolation tool operates via the HSA Tools API (not LD_PRELOAD) and intercepts:

- `hsa_queue_create`
- `hsa_amd_queue_intercept_create`
- `hsa_amd_queue_intercept_register`
- Executable symbol queries

Captured artifacts:

```
isolate_capture/
    dispatch.json
    kernarg.bin
    kernel.hsaco
    memory_regions.json
    memory/region_<base>.bin
```

The tool snapshots:

- Kernel dispatch metadata (grid/workgroup)
- Private/group segment sizes
- HSACO code object
- Full device memory state prior to dispatch

Thread safety is enforced via explicit locking. No dynamic allocation occurs inside low-level interceptors.

---

## 8.2 VM Diagnostic Tool

CLI:

```
rocm-perf replay reserve-check
```

This tool validates whether the current GPU supports deterministic fixed-address VM reservation.

Properties:

- Page-aligned reservation
- Strict equality check on returned base address
- Structured summary output
- Exit codes:
  - `0` → exact reservation success
  - `1` → reservation failure
  - `2` → relocation detected

CDNA-class GPUs (e.g., MI325 / gfx942) support deterministic reservation.
Some APUs (e.g., gfx1035) reject arbitrary fixed-address reservations.

---

## 8.3 Full VA-Faithful Replay

CLI:

```
rocm-perf replay full-vm
rocm-perf replay full-vm --iterations N
rocm-perf replay full-vm --iterations N --no-recopy
```

Arguments:

- `--iterations N` — replay the captured dispatch N times without rebuilding VM state.
- `--no-recopy` — skip restoring device memory between iterations (stateful replay).

Replay guarantees (default mode, no `--no-recopy`):

- ISA validation against captured metadata
- Page-aligned `hsa_amd_vmem_address_reserve`
- Exact-address enforcement (abort on relocation)
- Handle creation, mapping, access rights setup
- Offset-correct `hsa_memory_copy`
- Memory reconstruction before executable load
- Correct AQL packet construction
- Completion signal synchronization
- Original grid/workgroup dimensions restored from `dispatch.json`

Replay aborts immediately if:

- Any reservation fails
- Reserved base ≠ requested aligned base
- Handle/map/access/copy fails
- ISA mismatch occurs

### Multi-Iteration Semantics

In multi-iteration mode:

- VM reservation and mapping occur once.
- A single completion signal is reused and reset to `1` before each dispatch.
- The AQL packet is rewritten for every iteration.

Behavior differs depending on flags:

**Default (`--iterations N` without `--no-recopy`):**

- Device memory is recopied from the isolation snapshot before each iteration.
- Each dispatch sees an identical device memory state.
- Execution is deterministic if the kernel itself is deterministic.

**Stateful mode (`--iterations N --no-recopy`):**

- Memory is restored only once before the first iteration.
- Subsequent iterations operate on the mutated device state.
- Useful for stress testing and throughput benchmarking.

Replay reports:

- Iteration count
- Whether memory recopy is enabled
- Average GPU time per iteration
- Total wall-clock time

This produces deterministic cross-process kernel reproduction on CDNA-class GPUs (default mode) and a controlled stateful execution harness when `--no-recopy` is enabled.

---

# 9. Validation Context

Validated on:

- AMD Instinct MI300X (gfx942)
- AMD Instinct MI325 (gfx942)
- ROCm 7.1
- rocprofv3

Closed-loop optimization validated end-to-end on real hardware.

---

Architecture is designed to parameterize device ceilings and vector widths for future AMD GPU generations.
