# ARCHITECTURE

## High-Level Flow

```
Application → Profiling → Roofline → DAG → ATT → Classification → Optimization Loop
```

---

## 1. Profiling Layer

Uses rocprofv3 for gfx942 hardware counters.

Data captured:
- Kernel durations
- FLOP counts
- Memory transactions
- Occupancy

Normalized into internal metric representation.

---

## 2. Roofline Layer

Computes:

Performance = FLOPs / time
Operational Intensity = FLOPs / bytes

Compared against:
- Peak FP throughput (MI300X)
- Peak HBM bandwidth

Determines performance regime.

---

## 3. DAG Engine

Graph constructed from:
- HIP stream dependencies
- Synchronization events

Critical path computed via longest path algorithm.

Outputs per-kernel impact weight.

---

## 4. ATT Deep Layer

Parses ATT traces to extract:
- Stall cycles
- Issue slot utilization
- Wave occupancy

Feeds bottleneck classifier.

---

## 5. Optimization Engine

Two-layer guard system:

### Static Guards
- Signature invariance
- AST verification

### Dynamic Guards
- Successful compilation
- Performance improvement
- Optional numeric validation

---

## 6. Closed-Loop Control

Optimization loop continues until:
- No further improvement
- Candidate pool exhausted
- Time budget exceeded

Rollback guaranteed on regression.

---

## Validation Context

Validated on:
- AMD Instinct MI300X
- gfx942
- ROCm 6.x

Architecture designed for extensibility to future AMD GPUs.
