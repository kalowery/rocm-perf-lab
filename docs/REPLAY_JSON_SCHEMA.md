# Replay JSON Schema

This document defines the structured JSON output contract for:

```
rocm-perf replay full-vm --json
```

This schema is intended for:

- CI systems
- Automated benchmarking
- LLM optimization loops
- Regression tracking
- Performance dashboards

When `--json` is enabled, replay emits **only valid JSON to stdout**.

---

# 1. Example Output

```json
{
  "kernel": {
    "name": "pointer_chase(Node*, int*)",
    "hsaco_path": "isolate_capture_1485832/kernel.hsaco"
  },
  "execution": {
    "iterations": 20,
    "mode": "stateless"
  },
  "timing": {
    "unit": "microseconds",
    "average": 3.12,
    "min": 2.88,
    "max": 3.44
  },
  "environment": {
    "gpu_agent": "gfx942",
    "rocm_version": "1.18",
    "pid": 1491806
  }
}
```

---

# 2. Top-Level Structure

Replay JSON always contains the following top-level keys:

```
kernel	execution	timing	environment
```

All fields described below are guaranteed to be present.

---

# 3. Field Definitions

## 3.1 kernel

### kernel.name

**Type:** string  
**Description:**
Demangled kernel symbol name captured during isolation.

Normalization rules:

- Backend clone annotations (e.g. `[clone .kd]`) are stripped.
- If demangled name is unavailable, the mangled name is used.

---

### kernel.hsaco_path

**Type:** string  
**Description:**
Filesystem path of the HSACO used during replay.

This reflects:

- The capture directory HSACO (default), or
- The override path provided via `--hsaco`.

This field ensures replay provenance and reproducibility.

---

## 3.2 execution

### execution.iterations

**Type:** integer  
**Description:**
Number of kernel dispatch iterations executed during replay.

---

### execution.mode

**Type:** string  
**Allowed Values:**

- `"stateless"`
- `"stateful"`

**Description:**

- `stateless` → Device memory is restored from the captured snapshot before each iteration.
- `stateful` → Memory is not restored between iterations (`--no-recopy`). Kernel mutations accumulate.

---

## 3.3 timing

All timing values are derived from GPU hardware timestamps via the HSA profiling API.

Host wall-clock time is not used.

### timing.unit

**Type:** string  
**Current Value:** `"microseconds"`

Future-proofed for potential alternative units.

---

### timing.average

**Type:** float  
**Description:**
Mean GPU execution time across all iterations.

---

### timing.min

**Type:** float  
**Description:**
Minimum observed GPU execution time.

---

### timing.max

**Type:** float  
**Description:**
Maximum observed GPU execution time.

---

## 3.4 environment

### environment.gpu_agent

**Type:** string  
**Description:**
Name of the GPU agent executing replay (e.g., `gfx942`).

---

### environment.rocm_version

**Type:** string  
**Description:**
ROCm runtime version reported by HSA (`major.minor`).

---

### environment.pid

**Type:** integer  
**Description:**
Process ID of the replay process.

Useful for correlating logs or system traces.

---

# 4. Determinism and Stability Guarantees

Replay JSON guarantees:

- All documented fields are always present.
- Field names will not change within a minor release.
- Numeric values are computed exclusively from GPU timestamps.

Replay JSON does NOT guarantee:

- Bitwise-identical floating point formatting across platforms.
- Identical timing results across different GPUs or ROCm versions.

---

# 5. Schema Evolution Policy

Future extensions may add:

- Grid/block dimensions
- Raw timing arrays (`raw_ns`)
- Strict replay metadata block
- Baseline comparison block

New fields may be added without breaking backward compatibility.

Existing fields will not be removed or renamed without a major version bump.

---

# 6. Example JSON Validation Snippet (Python)

```python
import json

with open("replay_output.json") as f:
    data = json.load(f)

assert "kernel" in data
assert "timing" in data
assert data["execution"]["iterations"] > 0
assert data["timing"]["average"] > 0
```

---

# 7. Intended Usage Patterns

Replay JSON is designed for:

- Automated accept/reject loops
- CI performance regression gates
- Benchmark dashboards
- LLM optimization scoring

It is not intended for human-readable reporting.

---

End of document.
