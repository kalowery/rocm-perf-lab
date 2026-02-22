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

### 8. Kernel Isolation & VA-Faithful Replay
- HSA Tools API–based kernel interception (no LD_PRELOAD hacks)
- Captures HSACO, kernarg, dispatch geometry, and full device memory snapshot
- Reconstructs original GPU virtual address layout using AMD VM APIs
- Deterministic cross-process kernel reproduction on CDNA-class GPUs
- Multi-iteration replay harness with optional stateful execution

Replay CLI:

```bash
rocm-perf replay reserve-check
rocm-perf replay full-vm
rocm-perf replay full-vm --iterations 100
rocm-perf replay full-vm --iterations 100 --no-recopy
```

- `reserve-check` validates fixed-address VM feasibility.
- `full-vm` performs strict VA-faithful replay.
- `--iterations N` repeats dispatch N times without rebuilding VM state.
- `--no-recopy` skips restoring memory between iterations (stateful mode).

---

## Build & Installation

rocm-perf-lab now has two components:

1. **Python CLI + analysis framework** (installed via `pip`)
2. **Native C++ isolation & replay tools** (built via CMake)

Both are required for full functionality.

---

### 1. System Requirements

- ROCm toolchain (validated on ROCm 7.1)
- `hipcc` available in PATH
- `rocprofv3` available in PATH
- CDNA-class GPU recommended for full VM-faithful replay (MI300X / MI325)

---

### 2. Install Python Package

From the project root:

```bash
pip install -e .
```

For development (includes pytest):

```bash
pip install -e .[dev]
```

This installs the CLI command:

```bash
rocm-perf
```

---

### 3. Build Native Replay Tools

The replay subsystem requires building the C++ components:

```bash
cd rocm_perf_lab/replay
mkdir -p build
cd build
cmake ..
make -j
```

This produces:

- `rocm_perf_replay_full_vm`
- `vm_reserve_only`

These binaries are invoked by the CLI via:

```bash
rocm-perf replay full-vm
rocm-perf replay reserve-check
```

---

## Quickstart (MI300X / gfx942)

### 1. Profile

### 2. Profile
```bash
rocm-perf profile ./your_app
```

Generates:
- `profile.json`
- `roofline.json`
- `dag.json`
- `analysis.json`

### 3. Optimize (Closed Loop)
```bash
rocm-perf optimize ./your_app
```

The optimizer:
1. Profiles kernels
2. Ranks by critical-path impact
3. Generates guarded HIP rewrites
4. Rebuilds and re-measures
5. Accepts only validated improvements

---

### 4. Kernel Isolation & Replay Workflow

1. Run application with isolation tool enabled (HSA Tools API).
2. Snapshot is written to:
   ```
   rocm_perf_lab/isolate/tool/isolate_capture/
   ```
3. Validate VM feasibility:
   ```bash
   rocm-perf replay reserve-check
   ```
4. Perform deterministic replay:
   ```bash
   rocm-perf replay full-vm
   ```
5. Run as microbenchmark:
   ```bash
   rocm-perf replay full-vm --iterations 100
   ```

Replay reports average GPU time and total wall time.

---

## Running Integration Tests (MI325 / CDNA GPUs)

The repository includes end‑to‑end integration tests that validate:

- Kernel isolation via HSA Tools API
- Virtual address–faithful replay
- Device memory snapshot and reconstruction
- Pointer‑dependent memory correctness (pointer‑chase test)

These tests must be run on a ROCm system with a supported CDNA GPU (e.g., MI300X / MI325).

### 1. Activate Python Environment

```bash
cd /work1/amd/klowery/workspace/rocm-perf-lab
source /work1/amd/klowery/workspace/venv/bin/activate
```

### 2. Pull Latest Changes

```bash
git pull
```

### 3. Install Editable Package

```bash
pip install -e .
```

### 4. Build Native Components

Build isolate tool:

```bash
cd rocm_perf_lab/isolate/tool
mkdir -p build
cd build
cmake ..
make -j
```

Build replay tools:

```bash
cd /work1/amd/klowery/workspace/rocm-perf-lab/rocm_perf_lab/replay
mkdir -p build
cd build
cmake ..
make -j
```

### 5. Run Integration Tests

```bash
cd /work1/amd/klowery/workspace/rocm-perf-lab
pytest tests/integration -v
```

The current integration suite includes:

- `test_minimal_isolate_and_replay` – simple arithmetic kernel
- `test_pointer_chase_isolate_and_replay` – pointer‑chasing memory validation kernel

Both tests:
- Build a HIP test kernel
- Run it with the isolate tool enabled
- Validate snapshot creation
- Perform deterministic VA‑faithful replay

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
