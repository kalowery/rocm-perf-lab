# Roofline Design Notes — rocm-perf-lab

This document explains how roofline analysis is implemented.

---

# 1. Goals

Provide a minimal, stable roofline model that:

- Uses rocprof hardware counters
- Computes FP32 FLOPs
- Computes DRAM bytes
- Classifies memory vs compute bound
- Never breaks profiling if counters unavailable

---

# 2. Metrics Used

Requested metrics:

- SQ_INSTS_VALU
- TCC_READ_BYTES
- TCC_WRITE_BYTES

FLOPs estimate:

```
FLOPs ≈ 2 × SQ_INSTS_VALU
```

Assumes FMA = 2 FLOPs.

Bytes:

```
bytes = TCC_READ_BYTES + TCC_WRITE_BYTES
```

---

# 3. Arithmetic Intensity

```
AI = FLOPs / bytes
```

If bytes = 0 → AI = 0.

---

# 4. Achieved Metrics

```
achieved_gflops = FLOPs / runtime
achieved_bandwidth = bytes / runtime
```

---

# 5. Bound Classification

Let:

- peak_compute = theoretical_peak_flops
- peak_bandwidth = peak_bandwidth_gbps

If:

```
peak_bandwidth × AI < peak_compute
```

Then:

```
bound = "memory"
```

Else:

```
bound = "compute"
```

---

# 6. Failure Handling

If:

- Metrics unavailable
- Counter parsing fails
- rocprof errors

Then:

```
"roofline": null
```

Profiling must still succeed.

---

# 7. Limitations

- FP32-only estimation
- Assumes VALU maps to FP32 arithmetic
- Does not distinguish FP16/FP64
- Does not model cache-level traffic

---

# 8. Future Extensions

Possible improvements:

- Separate FP16/FP32 counters
- Add L2 bandwidth modeling
- Integrate roofline plotting
- Add counter auto-detection

---

Roofline integration is intentionally minimal and robust.
