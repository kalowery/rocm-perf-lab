# Profiling Internals

## Kernel Selection

1. Query all dispatches from rocpd SQLite
2. Sort by duration
3. Filter runtime helpers:
   - `__amd_rocclr_*`
   - `hip*`
   - `hsa_*`
4. Select longest remaining dispatch

Fallback to longest dispatch if no compute kernel found.

---

## Resource Extraction

From `rocpd_info_kernel_symbol_*`:

- `arch_vgpr_count` → VGPR per thread
- `sgpr_count` → SGPR per wave
- `group_segment_size` → LDS per block

---

## Occupancy Modeling

Inputs:

- VGPR per thread
- LDS per block
- Threads per block
- Architecture limits

Occupancy = minimum of resource constraints.

This is theoretical occupancy only.

---

## Stability Model

Coefficient of variation (CV):

```
CV = stddev / mean
```

Classification:

- ≤ 0.05 → stable
- ≤ 0.10 → moderate
- > 0.10 → unstable

Repeat unstable runs under controlled conditions.
