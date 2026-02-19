# DEVELOPER GUIDE

## System Overview

rocm-perf-lab consists of the following subsystems:

1. Profiling Engine (rocprofv3 integration)
2. Roofline Analyzer (gfx942)
3. Critical-Path DAG Engine
4. ATT Deep Analysis Module
5. Bottleneck Classifier
6. Guarded HIP Optimizer
7. LLM Closed-Loop Controller

---

## Profiling Engine

Uses rocprofv3 counters specific to gfx942.

Key metrics:
- VALU/MFMA instruction counts
- LDS usage
- Global memory transactions
- Wave occupancy

Counters are normalized into unified metric schema.

---

## Roofline Module

Inputs:
- FLOP counters
- DRAM bytes
- Kernel time

Outputs:
- Operational intensity
- Achieved performance
- Bound classification

Ceilings are parameterized by device model (MI300X default).

---

## DAG Engine

Constructs graph:
- Nodes: kernels
- Edges: synchronization + stream dependencies

Computes:
- Longest path (critical path)
- Slack per node
- Global speedup potential

---

## ATT Integration

ATT traces provide:
- Wave state transitions
- Stall reasons
- Cache miss patterns

Feature extractor feeds classifier.

---

## Bottleneck Classifier

Feature inputs:
- Roofline distance
- Occupancy
- Stall breakdown
- Divergence metrics

Outputs single bottleneck label.

---

## Guarded HIP Optimizer

Constraints:
- Function signature identical
- No new global state
- No unsafe casts

Validation steps:
1. AST parse
2. Signature check
3. Recompile
4. Benchmark

---

## LLM Controller

Closed-loop flow:

1. Select kernel (critical path weighted)
2. Generate transformation proposal
3. Enforce signature invariants
4. Compile
5. Repair on error
6. Benchmark
7. Accept if improvement > threshold

All decisions logged in optimization trace JSON.

---

## Extending the System

To add new analysis modules:
- Register metric extractor
- Extend JSON schema
- Add classifier feature mapping

Ensure consistency across USER_GUIDE and ARCHITECTURE docs.
