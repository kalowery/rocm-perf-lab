# Replay Engine

The replay subsystem of **rocm-perf-lab** provides deterministic, VA-faithful reproduction of a captured GPU kernel dispatch. It enables hardware-accurate microbenchmarking and forms the foundation for replay-based LLM optimization workflows.

---

# 1. Purpose

Replay allows a single kernel dispatch from a large application to be:

- Captured once using the isolate tool
- Reconstructed in a clean process
- Re-executed deterministically
- Timed using GPU hardware timestamps
- Iterated rapidly without re-running the full application

This transforms a production workload into a standalone, reproducible kernel microbenchmark.

---

# 2. Isolation → Replay Workflow

## Step 1: Capture a Dispatch

Run the application with isolation enabled:

```bash
export HSA_TOOLS_LIB=rocm_perf_lab/isolate/tool/build/librocm_perf_isolate.so
export ISOLATE_KERNEL=<kernel_name>
export ISOLATE_DISPATCH_INDEX=0

./your_application
```

This produces:

```
isolate_capture_<pid>/
```

Containing:

- `kernel.hsaco`
- `dispatch.json`
- `kernarg.bin`
- `memory_regions.json`
- region memory blobs

---

## Step 2: Validate VM Feasibility

```bash
rocm-perf replay reserve-check --capture-dir isolate_capture_<pid>
```

Ensures fixed-address virtual memory reservation is possible.

---

## Step 3: Deterministic Replay

```bash
rocm-perf replay full-vm --capture-dir isolate_capture_<pid>
```

Optional iteration mode:

```bash
rocm-perf replay full-vm \
  --capture-dir isolate_capture_<pid> \
  --iterations 100
```

Stateful execution:

```bash
--no-recopy
```

---

# 3. Strict VA-Faithful Replay

Replay reconstructs the original GPU virtual address layout using:

- `hsa_amd_vmem_address_reserve`
- `hsa_amd_vmem_map`
- Fixed-address reservations

If any region relocates, replay aborts.

To avoid ROCr SVM aperture conflicts, replay uses deterministic pre-mmap steering before `hsa_init()`.

Replay guarantees:

- Identical virtual address layout
- Identical memory snapshot
- Identical kernarg blob
- Identical grid/block geometry
- GPU hardware timestamp timing

Replay does *not* guarantee:

- Cross-kernel scheduling equivalence
- Cache warmth equivalence
- Full application-level interaction fidelity

Replay validates the kernel in isolation.

---

# 4. JSON Output Mode

Replay supports structured output:

```bash
rocm-perf replay full-vm \
  --capture-dir isolate_capture_X \
  --iterations 20 \
  --json
```

When `--json` is enabled:

- Only valid JSON is emitted
- No human-readable text is printed

See `README.md` for field definitions.

---

# 5. HSACO Override

Replay can substitute a different kernel binary:

```bash
rocm-perf replay full-vm \
  --capture-dir isolate_capture_X \
  --hsaco ./optimized_kernel.hsaco
```

This enables:

- Replaying optimized kernels
- Comparing LLM-generated variants
- Kernel microbenchmarking without full application runs

The JSON output includes:

```
"hsaco_path": "..."
```

for provenance.

---

# 6. Replay in LLM Optimization

Replay enables a fast optimization loop:

```
1. Capture once
2. Generate optimized kernel
3. Rebuild kernel
4. Replay optimized HSACO
5. Compare timing against baseline
6. Accept/reject
```

This avoids expensive end-to-end application profiling for each candidate.

Replay-based validation is safe if:

- Kernel signature unchanged
- Grid/block unchanged
- Memory contract unchanged

`rocm-perf optimize` enforces signature stability.

---

# 7. Performance Characteristics

Replay typically runs in milliseconds, even if the full application runs in minutes.

Expected speedup for optimization loops:

- 100× to 1000× iteration speed increase

---

# 8. Recommended Usage Patterns

## Benchmarking

Use stateless multi-iteration replay for stable averages:

```bash
--iterations 50
```

## Optimization

Use:

```bash
--hsaco optimized.hsaco --json
```

Compare against stored baseline JSON.

## Debugging

Use stateful mode:

```bash
--no-recopy
```

To observe mutation accumulation.

---

# 9. Limitations

Replay is kernel-scoped.

It does not model:

- Kernel overlap
- Inter-kernel interference
- Host scheduling jitter
- System-level contention

Replay is ideal for kernel-level evolution, not full application validation.

---

# 10. Design Philosophy

Replay extracts deterministic kernel microbenchmarks from production workloads.

This enables:

- Reproducible hardware-timed measurements
- Rapid LLM-driven iteration
- Controlled experimentation
- CI-friendly performance validation

Replay transforms rocm-perf from a profiler into a kernel evolution platform.

---

End of document.
