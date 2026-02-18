# ROCm Perf Lab — Production Engineering Manual

This document is a comprehensive operational guide for engineers using `rocm-perf-lab` in production, research, benchmarking, or optimization workflows.

It assumes familiarity with:

- ROCm
- HIP kernels
- GPU performance concepts
- Basic Linux command line usage

---

# 1. System Overview

`rocm-perf-lab` is a deterministic, hardware-aware performance engineering toolkit for AMD GPUs running ROCm 7.x.

It provides:

- Structured kernel profiling via `rocprofv3`
- rocpd SQLite parsing (stable in ROCm 7.2+)
- Kernel metadata extraction (VGPR, SGPR, LDS)
- Theoretical occupancy modeling
- Stability analysis (coefficient of variation)
- Regression-based autotuning with pruning guarantees
- Optional counter-based roofline analysis
- Schema-validated JSON output

Design principles:

- Deterministic behavior
- Explicit hardware modeling
- Clean CLI semantics
- Fail-safe degradation

---

# 2. Supported Environment

Validated against:

- ROCm 7.2.0+
- RDNA2 (gfx1030) and CDNA2
- `rocprofv3`
- Linux (POSIX)
- Python 3.9–3.12

---

# 3. Installation

## Development / Editable Install

```bash
python -m venv venv
source venv/bin/activate
pip install -e .
```

Verify:

```bash
rocm-perf --version
```

---

# 4. Profiling Workflow

## 4.1 Standard Kernel Profiling

```bash
rocm-perf profile ./kernel_binary
```

Output includes:

- Mean runtime (ms)
- Coefficient of variation (CV)
- Theoretical occupancy
- Hardware resource allocation

This uses:

```
rocprofv3 --kernel-trace
```

Internally parsed from rocpd SQLite database.

---

## 4.2 JSON Output (Recommended for Automation)

```bash
rocm-perf profile ./kernel_binary --json
```

Example output:

```json
{
  "schema_version": "1.0",
  "kernel": {...},
  "gpu": {...},
  "runtime_ms": 0.012,
  "stability": {...},
  "resources": {...},
  "occupancy": {...},
  "roofline": null
}
```

JSON is validated against a strict Pydantic schema.

---

## 4.3 Quiet Mode (CI / Scripting)

```bash
rocm-perf profile ./kernel_binary --quiet
```

Outputs only numeric runtime.

Use this for regression tracking scripts.

---

## 4.4 Debug Mode

```bash
rocm-perf profile ./kernel_binary --debug
```

Shows raw `rocprofv3` logs.

Use when:

- rocprof fails
- Kernel metadata missing
- Counters unavailable

---

## 4.5 Timing-Only Mode

```bash
rocm-perf profile ./kernel_binary --no-rocprof
```

Measures entire process runtime.

Use when:

- rocprof unstable
- Hardware counters unnecessary
- Profiling full application instead of kernel

---

# 5. Stability Analysis

Coefficient of variation:

```
CV = stddev / mean
```

Classification:

| CV        | Classification |
|-----------|---------------|
| ≤ 0.05    | Stable        |
| ≤ 0.10    | Moderate      |
| > 0.10    | Unstable      |

If unstable:

- Reduce system noise
- Disable DVFS
- Increase runs
- Ensure thermal stability

---

# 6. Occupancy Modeling

Occupancy computed from:

- VGPR per thread
- LDS per block
- Threads per block
- Architecture constraints

Example:

```json
"occupancy": {
  "theoretical": 0.75,
  "threads_per_block": 256,
  "wave_size": 32
}
```

Important:

- This is a theoretical upper bound.
- It does not model memory stalls.
- High occupancy does not guarantee high performance.

---

# 7. Roofline Analysis (Optional)

Enable:

```bash
rocm-perf profile ./kernel_binary --roofline
```

Optional override:

```bash
--memory-bandwidth-gbps 100
```

Metrics used:

- SQ_INSTS_VALU
- TCC_READ_BYTES
- TCC_WRITE_BYTES

Computed values:

- FLOPs (FP32 estimate)
- Bytes moved
- Arithmetic intensity
- Achieved GFLOPs
- Achieved bandwidth
- Compute vs memory bound classification

If counters unavailable:

```json
"roofline": null
```

---

# 8. Autotuning Workflow

## 8.1 Basic Usage

```bash
rocm-perf autotune \
  --space search_space.json \
  --cmd-template "./kernel --bm {BLOCK_M} --bn {BLOCK_N}"
```

---

## 8.2 Algorithm Phases

1. Seed Phase — profile subset
2. Prediction Phase — regression on static features
3. Confirm Phase — profile pruned configs

Guarantee:

- No full search profiling
- Static-feature-only pruning

---

## 8.3 When R² Is Low

If:

```
"warning": "low_model_confidence"
```

Then:

- Increase seed fraction
- Reduce search dimensionality
- Inspect runtime variance

---

# 9. Reproducibility Checklist

For consistent results:

- Disable power scaling
- Lock GPU clocks (if possible)
- Ensure consistent thermal state
- Close background processes
- Repeat runs
- Use JSON output for logging

---

# 10. Troubleshooting

## rocprof fails

- Run with `--debug`
- Check ROCm version
- Validate kernel launches

## No kernel detected

- Ensure compute kernel dominates runtime
- Avoid tiny kernels

## Roofline null

- Hardware counters unavailable
- Unsupported metric group

---

# 11. Production Integration

For CI pipelines:

- Use `--quiet` for regression tracking
- Store JSON outputs for historical analysis
- Validate schema_version on ingestion

For research workflows:

- Record runtime, occupancy, roofline
- Report arithmetic intensity
- Report stability classification

---

# 12. Limitations

- Single dominant kernel assumption
- Theoretical occupancy only
- FP32-only roofline estimation
- No multi-kernel trace segmentation

---

# 13. Operational Philosophy

`rocm-perf-lab` is designed for:

- Explicit performance reasoning
- Deterministic measurement
- Reproducible optimization

It avoids hidden heuristics and silent behavior changes.

---

End of Production Engineering Manual.
