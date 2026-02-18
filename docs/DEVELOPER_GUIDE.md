# Developer Guide — rocm-perf-lab

This document explains the internal structure, invariants, and extension points of the project.

---

# 1. Project Structure

```
rocm_perf_lab/
├── analysis/          # Regression + feature engineering
├── autotune/          # Autotuning algorithm
├── cli/               # Typer CLI entrypoints
├── hal/               # Hardware abstraction layer
├── profiler/          # rocprof integration + pipeline
├── schema/            # Pydantic models (JSON contracts)
```

---

# 2. Core Invariants

These must not be broken:

1. Profile JSON must validate against schema.
2. Static features must be separate from runtime features.
3. Autotune must never profile all configs before pruning.
4. Profiling must degrade gracefully if rocprof fails.
5. JSON output must remain deterministic.

If changing schema structure:

- Bump schema_version.
- Update tests.
- Update whitepaper.

---

# 3. Profiling Pipeline

`build_profile()` is the central entrypoint.

Flow:

1. run_command()
2. Extract rocprof metadata
3. Compute occupancy
4. Optionally compute roofline
5. Validate schema

Do not bypass schema validation.

---

# 4. Hardware Abstraction Layer (HAL)

Add new architectures by:

1. Creating new class in `hal/`
2. Implementing:
   - compute_occupancy()
   - theoretical_peak_flops()
3. Registering in `registry.py`

Avoid hardcoding architecture values outside HAL.

---

# 5. Autotune Algorithm

Three phases:

1. Seed
2. Predict
3. Confirm

Pruning must use static features only.

Do not introduce runtime metrics into pruning.

---

# 6. Roofline Integration

Counter-based mode is optional.

Rules:

- Never break standard profiling if counters unavailable.
- Fail silently and set roofline = None.
- Do not assume metrics exist on all hardware.

---

# 7. Testing

Run:

```
pytest
```

Test coverage includes:

- Schema validation
- Stability classification
- Pruning logic
- CLI smoke tests

If adding features:

- Add unit tests.
- Ensure CI passes.

---

# 8. Release Workflow

1. Update version in:
   - pyproject.toml
   - __init__.py
2. Run tests.
3. Build wheel.
4. Tag release.
5. Upload via twine.

---

# 9. Extension Philosophy

This project prioritizes:

- Determinism over cleverness
- Correctness over micro-optimizations
- Explicit modeling over magic heuristics

Do not add fragile heuristics without strong justification.

---

# 10. Known Limitations

- Single dominant kernel assumption
- Theoretical occupancy only
- FP32-only roofline

Future work should extend carefully.
