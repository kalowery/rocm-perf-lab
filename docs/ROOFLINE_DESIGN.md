# Roofline Design — rocm-perf-lab

This document describes the **current** roofline implementation as validated on gfx942-class GPUs (MI300X / MI325) using ROCm 7.1 and rocprofv3.

The design goal is numerical correctness on CDNA3 while remaining robust to counter failures.

---

# 1. Design Goals

The roofline layer must:

- Use rocprofv3 hardware counters
- Produce architecture-aware FP32 FLOP estimates
- Compute DRAM traffic correctly on gfx942
- Classify memory vs compute bound regimes
- Fail gracefully if counters are unavailable

---

# 2. Counter Mapping (rocprofv3 / rocpd)

Metrics are resolved via the `rocpd_info_pmc` table (name → pmc_id mapping).

We explicitly avoid relying on ATT databases for roofline computation.

Required counters (gfx942):

- `SQ_INSTS_VALU`
- `MFMA_MOPS_F32` (if present)
- TCC read/write request counters (RDREQ / WRREQ variants)

---

# 3. FP32 FLOP Estimation (CDNA3-aware)

FLOPs are computed as:

```
FLOPs =
    SQ_INSTS_VALU * fp32_valu_width
  + MFMA_MOPS_F32 * 512
```

Where:

- `fp32_valu_width = 8` for CDNA3 (gfx942)
- MFMA contribution reflects matrix op throughput scaling

This corrects earlier scalar assumptions (e.g., 2 × VALU) and prevents systematic undercounting.

Limitations:
- FP32-focused model
- Does not currently model FP16/FP64 separately

---

# 4. DRAM Byte Accounting (gfx942)

We do **not** rely on `TCC_READ_BYTES` / `TCC_WRITE_BYTES` directly.

Instead we compute traffic from:

- RDREQ / WRREQ counters
- 32B vs 64B request variants

Total bytes:

```
bytes = (rdreq_32B * 32 + rdreq_64B * 64)
      + (wrreq_32B * 32 + wrreq_64B * 64)
```

This avoids incorrect aggregation and reflects actual memory traffic on CDNA3.

---

# 5. Arithmetic Intensity

```
AI = FLOPs / bytes
```

If `bytes == 0`, AI is defined as 0 to avoid division errors.

---

# 6. Achieved Performance Metrics

Runtime is computed from rocpd kernel dispatch timestamps:

```
runtime = SUM(dispatch_end - dispatch_start)
```

Derived metrics:

```
achieved_gflops     = FLOPs / runtime
achieved_bandwidth  = bytes / runtime
```

---

# 7. Bound Classification

Let:

- `peak_compute`
- `peak_bandwidth`

If:

```
peak_bandwidth * AI < peak_compute
```

Then kernel is classified as memory-bound; otherwise compute-bound.

This classification is first-order and feeds the bottleneck classifier.

---

# 8. Failure Semantics

If:

- Required counters are missing
- rocpd parsing fails
- rocprof execution fails

Then:

```
"roofline": null
```

Profiling must still succeed.

---

# 9. Validation Context

Validated on:

- AMD Instinct MI300X (gfx942)
- AMD Instinct MI325 (gfx942)
- ROCm 7.1
- rocprofv3

Numerical consistency verified against hardware measurements.

---

Roofline integration is architecture-aware for CDNA3 and designed to be extensible to future AMD GPUs via parameterized width and ceiling definitions.
