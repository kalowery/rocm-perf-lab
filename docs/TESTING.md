# TESTING GUIDE — rocm-perf-lab

This document describes testing practices for rocm-perf-lab.

Validated on gfx942-class GPUs (MI300X / MI325) using ROCm 7.1.

---

# 1. Running Tests Locally

Install development dependencies:

```
pip install -e .[dev]
```

Run unit tests:

```
pytest -q
```

Unit tests are hardware-independent unless explicitly marked.

---

# 2. Hardware-Independent Coverage

Covered areas include:

- JSON schema validation
- Bottleneck classification logic
- Optimization pruning logic
- CLI smoke tests
- Deterministic loop control logic

These tests must pass without ROCm hardware.

---

# 3. Hardware-Dependent Validation (Manual)

The following require gfx942 hardware and ROCm 7.1:

- rocprofv3 integration
- Counter mapping correctness
- Roofline FLOP scaling validation
- RDREQ / WRREQ byte accounting
- Critical path correctness
- Closed-loop optimization performance validation

These are not executed in CI.

---

# 4. Critical Path Validation Checklist

On hardware, verify:

- `.rocpd_profile` database is generated
- `critical_path_ns` is non-zero for multi-kernel workloads
- Dominant kernel matches expected runtime behavior
- ATT database is not used for runtime computation

Runtime must be computed from dispatch timestamps:

```
SUM(dispatch_end - dispatch_start)
```

---

# 5. Roofline Numerical Validation

On hardware, verify:

- VALU scaling matches CDNA3 expectation (`fp32_valu_width = 8`)
- MFMA contribution included (if present)
- Byte accounting uses RDREQ / WRREQ counters
- Operational intensity is consistent with expected kernel behavior

Small synthetic kernels (e.g., vector add) are recommended sanity checks.

---

# 6. Optimizer Stability Testing

When modifying optimization logic, validate:

- Kernel signature invariance enforcement
- Rejection path on regression
- Bounded repair loop (max 2 attempts)
- Deterministic baseline → patch → profile cycle
- Automatic rollback on failure

The optimizer must:

- Never produce ABI changes
- Never leave repository in broken build state
- Never accept regressions

---

# 7. CI Expectations

CI environment:

- Python 3.9–3.12
- Ubuntu (latest)

CI validates:

- Installation
- Unit tests
- Wheel build

CI does NOT validate GPU performance paths.

---

# 8. Schema Changes

If modifying profile JSON structure:

- Update relevant models
- Update unit tests
- Bump schema version (if applicable)
- Ensure backward compatibility if required

---

Testing discipline ensures deterministic behavior and prevents silent performance regressions.
