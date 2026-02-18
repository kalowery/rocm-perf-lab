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
