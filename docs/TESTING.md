# Testing Guide — rocm-perf-lab

This document explains how to test the project locally and in CI.

---

# 1. Running Tests Locally

Install dev dependencies:

```
pip install -e .[dev]
```

Run tests:

```
pytest -q
```

---

# 2. Test Coverage Areas

Current coverage includes:

- Schema validation
- Stability classification
- Pruning logic
- CLI version smoke test

These tests are hardware-independent.

---

# 3. Hardware-Dependent Features

The following require ROCm hardware to validate fully:

- rocprof integration
- Counter collection
- Roofline metrics

These are intentionally not unit-tested directly.

Instead, they must:

- Fail gracefully
- Never break schema
- Never crash without hardware

---

# 4. CI Expectations

CI runs on:

- Python 3.9–3.12
- Ubuntu latest

CI validates:

- Installation
- Tests pass
- Wheel builds successfully

---

# 5. Adding New Tests

When adding features:

- Add unit tests in `tests/`
- Ensure hardware-independent logic is tested
- Avoid requiring ROCm hardware in CI

---

# 6. Schema Changes

If modifying profile JSON structure:

- Update `ProfileModel`
- Update tests
- Bump schema_version
- Update VERSIONING.md

---

# 7. Debugging Failures

If schema validation fails:

- Inspect returned JSON
- Validate against ProfileModel
- Ensure optional fields handled correctly

---

Testing discipline is critical for stability.
