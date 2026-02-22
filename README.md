# rocm-perf-lab

**rocm-perf-lab** is a deterministic kernel analysis and closed-loop optimization framework for HIP applications on AMD Instinct GPUs (validated on **MI300X / MI325, gfx942**).

It combines:

- Roofline modeling
- Trace-derived critical-path DAG analysis
- ATT-based deep profiling
- A safety-constrained LLM optimizer
- Deterministic VA-faithful kernel replay

The core vision is to extract individual kernel dispatches from large production applications and convert them into hardware-accurate, reproducible microbenchmarks. These microbenchmarks can then be optimized rapidly using LLM-generated transformations without re-running the full application.

This turns rocm-perf-lab from a profiler into a **kernel evolution platform**.

---

## Closed-Loop Kernel Evolution via Deterministic Replay

Traditional optimization loop:

```
LLM rewrite → rebuild → profile full application → measure → accept/reject
```

Replay-enabled loop:

```
1. Capture kernel dispatch once (isolate)
2. Generate optimized kernel
3. Rebuild kernel
4. Replay optimized HSACO against captured memory state
5. Compare GPU timestamp timing
6. Accept or reject
```

Because replay reconstructs:

- Original GPU virtual addresses
- Full device memory snapshot
- Kernarg layout
- Dispatch geometry

The only variable during iteration is the kernel implementation itself.

Replay runs in milliseconds even when the full application runs in minutes, enabling 100×–1000× faster optimization cycles.

📘 See full replay documentation: [`docs/REPLAY.md`](docs/REPLAY.md)

---

## Why This Is Different

Most GPU optimization tools fall into one of two categories:

1. **Profilers** – They measure and report bottlenecks but do not provide a deterministic mechanism for validating transformations.
2. **Autotuners** – They search parameter spaces but typically operate on synthetic microbenchmarks detached from real production state.

rocm-perf-lab introduces a third model:

> Extract the *actual* kernel dispatch from a production application, reconstruct its exact memory and virtual address state, and turn it into a deterministic hardware-timed microbenchmark.

Key differentiators:

- ✅ Virtual-address-faithful replay (no relocation allowed)
- ✅ Full device memory snapshot restoration
- ✅ GPU hardware timestamp timing (not wall-clock)
- ✅ HSACO override for controlled kernel substitution
- ✅ Signature-constrained LLM rewrites
- ✅ JSON contract for CI automation

This enables a controlled kernel evolution workflow grounded in real workload state rather than synthetic benchmarks.

Replay is not a simulator and not an approximation. It is a strict reconstruction of a real dispatch, executed under the same hardware runtime conditions.

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

> 📘 See the full replay reference: [`docs/REPLAY.md`](docs/REPLAY.md)
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

## JSON Output (Replay)

The replay engine supports a structured JSON mode:

```bash
rocm-perf replay full-vm \
  --capture-dir isolate_capture_123456 \
  --iterations 5 \
  --json
```

When `--json` is specified, only valid JSON is emitted to stdout (no human-readable text).

### Example

```json
{
  "kernel": {
    "name": "pointer_chase(Node*, int*)"
  },
  "execution": {
    "iterations": 5,
    "mode": "stateless"
  },
  "timing": {
    "unit": "microseconds",
    "average": 3.06,
    "min": 2.88,
    "max": 3.44
  },
  "environment": {
    "gpu_agent": "gfx942",
    "rocm_version": "1.18",
    "pid": 1487354
  }
}
```

### Field Definitions

**kernel.name**  
Demangled kernel symbol name captured during isolation. Backend clone annotations (e.g., `[clone .kd]`) are stripped. Falls back to mangled name if necessary.

**execution.iterations**  
Number of kernel dispatch iterations executed during replay.

**execution.mode**  
- `"stateless"` – Device memory is restored from the captured snapshot before each iteration.
- `"stateful"` – Device memory is not restored between iterations (`--no-recopy`). Kernel mutations accumulate across iterations.

**timing.unit**  
Unit of timing statistics. Currently always `"microseconds"`.

**timing.average**  
Mean GPU execution time across all iterations.

**timing.min**  
Minimum observed GPU execution time.

**timing.max**  
Maximum observed GPU execution time.

Timing values are derived from GPU hardware timestamps via the HSA profiling API and reflect device execution time only (not host wall-clock time).

**environment.gpu_agent**  
Name of the GPU agent executing the replay (e.g., `gfx942`).

**environment.rocm_version**  
ROCm runtime version reported by the HSA runtime.

**environment.pid**  
Process ID of the replay process.

JSON mode is intended for CI systems, automated benchmarking, and regression tracking.

---

---

## Strict VA-Faithful Replay & ROCr SVM Aperture Steering

Deterministic replay depends on reproducing the **exact GPU virtual address (VA) layout** observed during the original application run. This includes not only individual allocations, but also the broader VA aperture topology established by ROCr.

The ability to reliably reconstruct these VA apertures is one of the key enablers of replay-driven kernel evolution.

### Background: ROCr SVM Aperture Placement

During `hsa_init()`, ROCr (via `libhsakmt`) reserves large CPU virtual address ranges to establish SVM apertures using:

```
mmap(PROT_NONE, MAP_ANONYMOUS | MAP_NORESERVE | MAP_PRIVATE)
```

The base of these apertures is chosen heuristically based on the current process VA layout. Small differences in layout can cause different aperture placements across runs.

Strict replay requires that captured GPU virtual addresses be reserved at their original fixed addresses using `hsa_amd_vmem_address_reserve()`.

If an ROCr SVM aperture overlaps a captured region:

- The reservation relocates, or
- Replay aborts (strict mode)

Empirically on MI325, this produced nondeterministic replay failures (~25% failure rate) even with identical capture data.

### Deterministic Steering Strategy

To preserve strict semantics while eliminating nondeterminism, `replay_full_vm` deliberately shapes the process VA layout *before* `hsa_init()`:

1. Parse captured memory regions before runtime initialization.
2. Temporarily `mmap(PROT_NONE | MAP_FIXED_NOREPLACE)` the captured VA ranges.
3. Call `hsa_init()`.
   - ROCr must choose SVM apertures that avoid these pre-mapped ranges.
4. `munmap()` the temporary placeholders.
5. Perform strict `hsa_amd_vmem_address_reserve()` at the original VAs.

This does **not** modify ROCr and does not relax strict replay semantics.
It ensures that ROCr’s heuristic aperture placement is steered away from captured ranges, making replay deterministic.

Observed on MI325:

- Before steering: 20 runs → 5 failures
- After steering: 20 runs → 0 failures

### Why This Matters

Replay is only meaningful if the kernel executes under the same virtual address topology as the original dispatch. Many GPU kernels encode pointer relationships and memory assumptions that depend on VA stability.

By deterministically reproducing ROCr’s aperture layout and preventing relocation:

- The kernel sees the same pointer values
- The same VA-dependent behavior is preserved
- LLM-generated kernel substitutions can be evaluated under identical memory topology

This strict VA fidelity is foundational to the replay-driven optimization vision described above.

Replay aborts on any relocation. There is no silent fallback.

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
