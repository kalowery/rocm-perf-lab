---
name: rocm-perf-lab
description: Hardware-aware ROCm kernel profiling and regression-based autotuning using rocprofv3. Use when profiling HIP/CUDA-like kernels on AMD GPUs, extracting VGPR/LDS/occupancy metrics, performing stability analysis, or running structured autotuning with pruning. Applies to ROCm 7.x environments using rocprofv3 and rocpd SQLite outputs.
---

# ROCm Perf Lab Skill

Use this skill when performing structured performance engineering on ROCm-based AMD GPUs.

This skill assumes:

- ROCm 7.x
- rocprofv3 available
- Single dominant kernel per execution

---

# Core Capabilities

1. Deterministic kernel profiling
2. Kernel metadata extraction (VGPR, SGPR, LDS)
3. Theoretical occupancy modeling
4. Stability analysis (CV classification)
5. Regression-based autotuning with pruning

---

# Profiling Workflow

## Basic Profiling

Run:

```
rocm-perf profile <binary>
```

For structured output:

```
rocm-perf profile <binary> --json
```

If debugging rocprof issues:

```
rocm-perf profile <binary> --debug
```

---

## When To Use `--no-rocprof`

Use `--no-rocprof` when:

- You only need coarse runtime
- rocprof is unstable
- Measuring full process runtime

Do NOT use `--no-rocprof` when hardware metrics are required.

---

# Interpreting Profile Output

Refer to: `references/profiling.md`

Key fields:

- runtime_ms
- stability.cv
- resources.vgpr_per_thread
- resources.lds_bytes
- occupancy.theoretical

If classification == "unstable", repeat profiling under controlled conditions.

---

# Autotuning Workflow

Use:

```
rocm-perf autotune \
  --space search_space.json \
  --cmd-template "./kernel --bm {BLOCK_M} ..."
```

Algorithm phases:

1. Seed phase (profile subset)
2. Prediction phase (static-feature regression)
3. Confirm phase (profile pruned configs)

If output contains:

```
"warning": "low_model_confidence"
```

Then regression pruning may be unreliable.

See: `references/autotune.md`

---

# Static vs Runtime Features

Static features (used for pruning):

- BLOCK_M
- BLOCK_N
- BLOCK_K
- num_warps
- num_stages
- threads_per_block

Runtime features (post-profile only):

- VGPR
- LDS
- Occupancy

Do not attempt to predict runtime features without profiling.

---

# Best Practices

- Ensure compute kernel dominates execution
- Avoid background load
- Use JSON mode for automation
- Validate R² before trusting pruning
- Repeat unstable measurements

---

# When Not To Use This Skill

Do not use when:

- Targeting NVIDIA CUDA
- ROCm < 7
- Multiple dominant kernels per execution
- Full system performance tracing required

---

# References

- CLI details → references/cli.md
- Profiling internals → references/profiling.md
- Autotune mechanics → references/autotune.md

Load references only when deeper detail is required.
