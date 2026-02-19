---
name: rocm-perf-lab
description: Multi-kernel GPU performance analysis and closed-loop optimization framework for ROCm 7.x using rocprofv3, rocpd, roofline modeling, ATT deep analysis, and dominance-aware LLM optimization.
---

# ROCm Perf Lab Skill (v2 — Multi-Kernel Aware)

Use this skill when performing structured GPU performance engineering on ROCm-based AMD GPUs.

Assumptions:
- ROCm 7.x
- rocprofv3 available
- rocpd SQLite dispatch database available
- HIP/CUDA-like GPU kernels
- Applications may contain multiple kernels per execution

---

# Core Capabilities

## 1. Deterministic GPU Runtime Modeling

- GPU runtime derived from rocpd kernel dispatch timestamps
- NOT host wall-clock time
- Supports multi-launch workloads
- Supports multi-kernel applications

---

## 2. Multi-Kernel Critical Path Analysis

- Reconstructs dispatch DAG from rocpd
- Identifies dominant kernel symbol
- Computes:
  - critical_path_ns
  - dominant_symbol
  - fraction (dominance fraction)

Dominance fraction:

    fraction = time(dominant_kernel) / total_critical_path_time

Whole-application speedup ceiling:

    1 / (1 - fraction)

---

## 3. Roofline Modeling

Extracts:
- FLOPs
- Bytes
- Arithmetic intensity
- Achieved GFLOPs
- Achieved GB/s
- Bound classification (memory vs compute)

Supports gfx942 (MI300X / MI325) counter model.

---

## 4. ATT Deep Analysis

Extracts:
- Instruction mix (VALU, SALU, VMEM, LDS, MFMA, etc.)
- Stall fraction
- Idle fraction
- IPC
- Average memory latency

Gracefully degrades if ATT parsing fails.

---

## 5. Headroom-Based Optimization (HBO)

Headroom fraction estimates microarchitectural inefficiency.

High headroom → latency-bound or pipeline inefficiency.
Low headroom → likely algorithmic bound.

---

## 6. Closed-Loop LLM Optimization

Command:

    rocm-perf llm-optimize <source.cu> "<binary>" --auto-approve

Features:
- Strict fenced C++ patch contract
- Kernel signature preservation
- Dominant-kernel targeting
- Dominance-shift detection
- Iterative refinement
- Whole-application regression gating
- Architectural regression detection

---

## 7. Dominance-Aware Multi-Kernel Optimization

If:

    critical_path.fraction < 0.7

Then:
- Whole-app ceiling limited
- Expect dominance shifts
- Iteratively optimize top kernels

Dominance shifts automatically retarget optimization.

---

# CLI Overview

See references/cli.md
