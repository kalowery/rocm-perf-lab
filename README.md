# rocm-perf-lab

**rocm-perf-lab** is a performance analysis and closed-loop optimization framework for HIP applications on AMD Instinct GPUs, validated on **MI300X (gfx942)**. It combines roofline modeling, critical-path DAG analysis, ATT-based deep profiling, and an LLM-driven guarded optimizer to deliver measurable performance gains with safety guarantees.

---

## Key Capabilities

### 1. Roofline Integration (gfx942 / MI300X)
- Uses **rocprofv3 hardware counters** on gfx942
- Computes achieved FLOP/s and memory bandwidth
- Maps kernels onto device-specific roofline
- Identifies compute-bound vs memory-bound regimes

### 2. Critical-Path DAG Engine
- Builds a kernel-level execution DAG
- Computes the true critical path
- Quantifies slack and overlap potential
- Prioritizes optimizations by end-to-end impact

### 3. ATT-Based Deep Analysis
- Integrates **AMD Trace Tool (ATT)** data
- Wave occupancy, VALU/MFMA utilization
- Memory stalls, cache behavior
- ISA-level efficiency insights

### 4. Bottleneck Classifier
Classifies kernels into categories:
- Memory-bandwidth bound
- Latency bound
- Under-occupied
- Divergence limited
- Compute throughput limited

### 5. Extended Profiling JSON
Produces structured JSON including:
- Kernel metrics
- Roofline coordinates
- DAG criticality scores
- Bottleneck classification
- ATT-derived features

### 6. Guarded HIP Optimizer
- Applies constrained HIP transformations
- Signature enforcement (no ABI changes)
- Compile-time validation
- Runtime equivalence checks (optional)

### 7. LLM Closed-Loop Optimization Engine
- Generates candidate kernel transformations
- Enforces function signature invariants
- Compiler-repair loop for invalid outputs
- Accepts changes only if performance improves
- Automatic rollback on regressions

---

## Quickstart (MI300X / gfx942)

### 1. Build
```bash
cmake -S . -B build
cmake --build build -j
```

### 2. Profile
```bash
rocm-perf-lab profile ./your_app
```

Generates:
- `profile.json`
- `roofline.json`
- `dag.json`
- `analysis.json`

### 3. Optimize (Closed Loop)
```bash
rocm-perf-lab optimize ./your_app
```

The optimizer:
1. Profiles kernels
2. Ranks by critical-path impact
3. Generates guarded HIP rewrites
4. Rebuilds and re-measures
5. Accepts only validated improvements

---

## Safety Model

- No binary patching
- No unsafe pointer rewriting
- Function signatures preserved
- Deterministic build verification
- Automatic rollback on performance regression
- Optional numerical validation hooks

---

## Validation Context

All architectural assumptions and counter mappings are validated against:
- **AMD Instinct MI300X**
- **gfx942** architecture
- ROCm 6.x toolchain
- rocprofv3 counter interface

---

## Documentation

- `docs/USER_GUIDE.md` – How to use the tool
- `docs/DEVELOPER_GUIDE.md` – Extending and contributing
- `docs/ARCHITECTURE.md` – Internal system design
- `docs/WHITEPAPER.md` – Design rationale and methodology

---

## License

See LICENSE file.
