# ROCm Perf Lab — Technical Whitepaper

## Abstract

`rocm-perf-lab` is a deterministic, architecture-aware performance engineering toolkit for ROCm-based AMD GPUs. It combines structured kernel profiling, hardware resource extraction, occupancy modeling, and regression-based autotuning into a unified CLI-driven system.

This whitepaper explains the internal architecture, mathematical foundations, and design decisions behind the system.

---

# 1. Design Goals

The system was built around five core principles:

1. Determinism — identical inputs produce identical outputs.
2. Hardware awareness — incorporate architectural constraints.
3. Structured output — machine-consumable JSON first.
4. Separation of static vs runtime features.
5. Correctness before optimization.

---

# 2. Profiling Architecture

## 2.1 rocprofv3 Integration

`rocm-perf-lab` uses:

```
rocprofv3 --kernel-trace
```

Output is parsed from the `rocpd` SQLite database (not CSV).

### Why SQLite?

- Stable schema in ROCm 7.2
- Rich metadata (VGPR, SGPR, LDS)
- More reliable than CSV counters

---

## 2.2 Kernel Selection Strategy

Each run may contain multiple dispatches:

- Runtime helper kernels
- Memory operations
- Compute kernel

Selection algorithm:

1. Query all dispatches ordered by duration.
2. Filter runtime helpers:
   - `__amd_rocclr_*`
   - `hip*`
   - `hsa_*`
3. Select longest remaining dispatch.
4. Fallback to longest if no compute kernel found.

This avoids mistakenly profiling runtime memset/fill kernels.

---

# 3. Hardware Resource Extraction

From `rocpd_info_kernel_symbol_*`:

- `arch_vgpr_count` → VGPR per thread
- `sgpr_count` → SGPR per wave
- `group_segment_size` → LDS per block

These are architecture-level allocations, not runtime counters.

---

# 4. Occupancy Modeling

Occupancy is computed using a hardware abstraction layer (HAL).

Inputs:

- VGPR per thread
- LDS per block
- Threads per block
- Architecture limits

For RDNA2:

Occupancy = min(
    VGPR-limited waves,
    LDS-limited waves,
    hardware max waves
)

Returned as theoretical upper bound.

This does not model:

- Memory bandwidth limits
- Cache behavior
- Instruction-level stalls

---

# 5. Stability Modeling

Coefficient of variation (CV):

```
CV = stddev / mean
```

Classification:

- ≤ 5% → stable
- ≤ 10% → moderate
- > 10% → unstable

Unstable measurements warn in human mode.

---

# 6. Autotuning Architecture

## 6.1 Three-Phase Algorithm

1. Seed Phase
   - Profile subset
   - Build regression

2. Prediction Phase
   - Use static features only
   - Predict runtimes
   - Prune

3. Confirm Phase
   - Profile surviving configs
   - Select best

This guarantees no unnecessary profiling.

---

## 6.2 Feature Separation

Static (pre-profile):

- ACC
- ACC²
- BLOCK_K
- num_warps
- num_stages
- threads_per_block

Runtime (post-profile):

- VGPR
- LDS
- Occupancy

Pruning uses static features only to preserve correctness.

---

## 6.3 Regression Model

Polynomial regression (degree=2).

R² < 0.75 triggers warning.

This protects against unreliable pruning.

---

# 7. JSON Schema Guarantees

Profile output validated against Pydantic schema.

Ensures:

- Field presence
- Type correctness
- Version stability

Current schema version:

```
schema_version = "1.0"
```

---

# 8. Limitations

- Assumes single dominant kernel.
- Occupancy theoretical only.
- Regression model simple by design.
- No cross-kernel interaction modeling.

---

# 9. Future Directions

- Stability gating in autotune
- Caching repeated configurations
- Counter-based roofline integration
- Multi-kernel trace segmentation
- Packaging for PyPI release

---

# 10. Conclusion

`rocm-perf-lab` provides a principled foundation for performance engineering on ROCm. It bridges deterministic profiling with model-driven optimization while maintaining schema stability and architectural awareness.
