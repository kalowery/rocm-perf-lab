# Versioning

## v0.3-strict-replay-deterministic (2026-02-22)

### Highlights
- Eliminated nondeterministic failures in strict VA-faithful replay on CDNA GPUs (MI325).
- Implemented pre-mmap steering mechanism to avoid ROCr SVM aperture collisions during `hsa_init()`.
- Replay remains strictly VA-faithful and aborts on any relocation.
- Added detailed documentation in README and source explaining ROCr SVM behavior and deterministic solution.

### Impact
- Before: ~25% replay failure rate under repeated runs.
- After: 0% failures across repeated 20-run stress tests (minimal and pointer-chase kernels).

---

# Versioning Policy — rocm-perf-lab

`rocm-perf-lab` follows Semantic Versioning:

MAJOR.MINOR.PATCH

## While version < 1.0.0

The API and schema may evolve.

- MINOR may introduce breaking changes.
- PATCH is for bug fixes.

## After 1.0.0

- MAJOR: Breaking CLI or schema changes.
- MINOR: Backward-compatible feature additions.
- PATCH: Bug fixes only.

---

## Schema Version

Profile JSON output includes:

```
"schema_version": "1.0"
```

If the JSON structure changes incompatibly, increment:

- MAJOR version
- schema_version field

---

## Release Checklist

1. Update version in:
   - pyproject.toml
   - rocm_perf_lab/__init__.py
2. Run tests
3. Build wheel
4. Validate install
5. Tag release
6. Upload via twine

---

Use disciplined releases. Avoid silent schema drift.
