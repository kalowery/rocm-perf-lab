# USER GUIDE — rocm-perf-lab

rocm-perf-lab is a profiling and guarded optimization framework for HIP workloads targeting **gfx942-class GPUs (MI300X / MI325)**.

Validated on:

- AMD Instinct MI300X (gfx942)
- AMD Instinct MI325 (gfx942)
- ROCm 7.1
- rocprofv3

It combines architecture-aware roofline modeling, trace-derived critical path analysis, ATT deep inspection, and a safety-constrained closed-loop optimizer.

---

# 1. Profiling Workflow

## Baseline Profiling

```bash
rocm-perf profile ./app
```

This performs:

- Base rocprofv3 profiling (authoritative `.rocpd_profile`)
- Hardware counter collection
- Roofline computation
- Critical path extraction
- Optional ATT deep analysis pass

Outputs include:

- `profile.json`
- `roofline.json`
- `dag.json`
- `analysis.json`

Runtime and critical path are derived from dispatch timestamps in the base profile.

---

# 2. Worked Example: Profiling the Softmax Testbed

The repository includes small test workloads (e.g., a scaled softmax kernel) designed to exercise the profiling and optimization pipeline.

Assume you have built the softmax testbed binary:

```bash
hipcc softmax_testbed.cpp -O3 -o softmax_testbed
```

### Step 1 — Run Profiling

```bash
rocm-perf profile ./softmax_testbed
```

Profiling flow:

```
softmax_testbed
      │
      ▼
rocprofv3 (base pass)
      │
      ▼
.rocpd_profile (authoritative trace DB)
      │
      ├──► Runtime accounting (dispatch timestamps)
      │
      ├──► Roofline analysis (FLOPs + DRAM bytes)
      │
      └──► Trace-derived DAG (critical path)

(optional)
      │
      ▼
ATT pass ──► Stall + occupancy feature extraction
```

Internally, the following happens:

1. rocprofv3 runs and generates a `.rocpd_profile` database.
2. Kernel dispatch timestamps are extracted.
3. Hardware counters are resolved via `rocpd_info_pmc`.
4. FLOPs and DRAM bytes are computed.
5. Roofline placement is determined.
6. A trace-derived DAG is constructed.
7. (Optional) ATT pass enriches stall and occupancy metrics.

### Step 2 — Inspect Output

Example snippet from `analysis.json`:

```json
{
  "kernel": "softmax_kernel",
  "time_ms": 113.19,
  "operational_intensity": 0.18,
  "roofline_position": "memory_bound",
  "critical_path_score": 0.94,
  "bottleneck": "latency_bound"
}
```

Interpretation:

- The kernel lies on the critical path (score ≈ 1.0).
- Operational intensity is low.
- Classified as memory/latency bound.

This suggests that increasing arithmetic throughput alone will not help. Improving memory behavior or hiding latency is more promising.

---

# 3. Running Closed-Loop Optimization

```bash
rocm-perf-lab optimize ./softmax_testbed
```

Optimization cycle:

1. Baseline is profiled (authoritative `.rocpd_profile`).
2. Kernel is ranked by critical-path contribution.
3. A loop-unrolling proposal is generated.
4. Static guards verify signature invariance.
5. `hipcc` rebuilds the binary.
6. Re-profiling measures actual runtime.
7. If runtime improves → change accepted.
   If not → rollback.

Example improvement log (illustrative):

```
Baseline runtime: 113.19 ms
Iteration 1 runtime: 18.02 ms
Improvement: 84%
Accepted.
```

All improvements are measured, not assumed.

---

# 4. Roofline Analysis

For each kernel, the system computes:

- FP32 FLOPs (CDNA3-aware scaling)
- DRAM bytes (RDREQ / WRREQ based)
- Operational intensity
- Achieved GFLOP/s
- Achieved bandwidth
- Memory vs compute bound regime

Roofline is first-order and feeds bottleneck classification.

See `ROOFLINE_DESIGN.md` for implementation details.

---

# 5. Critical Path View

The DAG engine constructs a trace-derived execution graph from dispatch data.

It computes:

- `critical_path_ns`
- Per-kernel slack
- Optimization impact weighting

Optimizing non-critical kernels yields limited end-to-end speedup.

---

# 6. Optimizer Scope (v1)

The optimizer is intentionally constrained.

Scope:

- Standalone HIP kernel source files (`.cu`)
- No ABI changes
- No whole-application refactoring

Allowed transformation (v1):

- Safe loop unrolling (factor 2–8)

Guardrails:

- No unrolling across `__syncthreads`
- No unrolling across atomics
- No unrolling across dynamic shared memory usage
- Wave64-aware
- Avoid excessive VGPR pressure

This ensures safety and determinism.

---

# 7. Safety Model

- No binary patching
- No unsafe pointer rewriting
- Kernel signatures preserved
- Deterministic rebuild required
- Performance regression automatically reverted

Only empirically validated improvements are retained.

---

---

# 8. Kernel Isolation & VA-Faithful Replay

rocm-perf-lab includes a deterministic kernel isolation and replay subsystem.

This enables capturing a single GPU kernel dispatch — including its full device memory state — and reproducing it in a separate process with identical virtual addresses.

## 8.1 Isolation Snapshot

When the isolation tool is enabled (HSA Tools API interception), it produces:

```
rocm_perf_lab/isolate/tool/isolate_capture/
    dispatch.json
    kernarg.bin
    kernel.hsaco
    memory_regions.json
    memory/region_<base>.bin
```

The snapshot contains:

- Original grid and workgroup dimensions
- Kernel object metadata
- HSACO code object
- Full device memory contents prior to dispatch
- Original virtual address layout

No pointer rewriting or relocation is performed during capture.

---

## 8.2 Validate VM Feasibility

Before replaying, validate that the current GPU supports fixed-address reservation:

```bash
rocm-perf replay reserve-check
```

This checks whether `hsa_amd_vmem_address_reserve` can reserve the captured addresses exactly.

Exit codes:

- `0` — exact reservation supported
- `1` — reservation failure
- `2` — relocation detected (address mismatch)

CDNA-class GPUs (e.g., MI300X / MI325) support deterministic reservation.
Some APUs may not.

---

## 8.3 Deterministic Replay

Basic replay:

```bash
rocm-perf replay full-vm
```

Replay performs:

1. VM reservation (page-aligned)
2. Memory mapping and access setup
3. Device memory reconstruction
4. HSACO load and executable creation
5. Correct AQL dispatch packet construction
6. Completion signal wait

If any VM step fails or relocation occurs, replay aborts.

---

## 8.4 Multi-Iteration Replay (Microbenchmark Mode)

Replay can be executed multiple times without rebuilding VM state:

```bash
rocm-perf replay full-vm --iterations 100
```

Behavior:

- VM reservation and mapping occur once.
- Memory is recopied from the snapshot before each iteration.
- Each iteration sees identical device memory state.
- Average GPU time per iteration is reported.

Example output:

```
Memory reconstructed.
Iterations: 100
Recopy between iterations: yes
Average GPU time: 0.131 ms
Total wall time: 13.4 ms
```

This mode is deterministic if the kernel itself is deterministic.

---

## 8.5 Stateful Replay Mode

Replay can optionally skip memory restoration between iterations:

```bash
rocm-perf replay full-vm --iterations 100 --no-recopy
```

Behavior:

- Memory is restored only once before the first iteration.
- Subsequent iterations operate on mutated device state.
- Useful for stress testing and throughput benchmarking.

In this mode, execution may diverge depending on kernel side effects.

---

## 8.6 Timing Semantics

Replay reports:

- Iteration count
- Whether memory recopy is enabled
- Average GPU time per iteration
- Total wall-clock time

Timing excludes VM setup and reflects dispatch + completion wait time.

---

For deeper technical details, see:

- `ARCHITECTURE.md`
- `DEVELOPER_GUIDE.md`
- `ROOFLINE_DESIGN.md`
