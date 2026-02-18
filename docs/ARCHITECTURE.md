# Architecture Overview — rocm-perf-lab

This document explains how the major subsystems interact and why they are structured this way.

---

# 1. High-Level Design

```
CLI → Profiling Pipeline → HAL → Schema Validation
                     ↘
                      Autotune Engine
                     ↘
                      Roofline (optional)
```

The system is deliberately layered:

- CLI layer is thin.
- Pipeline layer orchestrates.
- HAL encapsulates hardware specifics.
- Schema layer enforces contracts.

No layer should bypass the one below it.

---

# 2. Profiling Subsystem

Located in `profiler/`.

Responsibilities:

- Invoke rocprofv3
- Parse rocpd SQLite output
- Identify dominant compute kernel
- Extract resource metadata
- Compute occupancy
- Optionally compute roofline

Key principle:

> Parsing is deterministic and architecture-neutral.

Architecture-specific math is delegated to HAL.

---

# 3. Hardware Abstraction Layer (HAL)

Located in `hal/`.

Purpose:

- Encapsulate architectural limits
- Provide occupancy model
- Provide theoretical peak FLOPs
- Provide peak bandwidth estimate

The profiler must not embed hardware constants directly.

---

# 4. Autotune Engine

Located in `autotune/`.

Three-phase algorithm:

1. Seed
2. Predict
3. Confirm

Critical invariant:

> Pruning uses static features only.

Runtime features are never used for prediction.

---

# 5. Schema Layer

Located in `schema/`.

Every profile JSON is validated before returning.

This guarantees:

- Backward compatibility
- Stable API surface
- Early failure on regressions

---

# 6. Roofline Subsystem

Counter-based optional extension.

Design principles:

- Fully optional
- No hard dependency on counters
- Graceful degradation
- Explicit FP32 assumption

If counters are unavailable:

```
"roofline": null
```

---

# 7. Failure Modes

The system must never crash due to:

- Missing counters
- rocprof instability
- Absent kernel metadata

Fallback behavior is mandatory.

---

# 8. Design Philosophy

The architecture favors:

- Determinism
- Explicit modeling
- Strict contracts
- Predictable CLI behavior

It avoids:

- Hidden heuristics
- Architecture hardcoding
- Silent schema drift

---

# 9. Evolution Strategy

Future features must:

- Respect schema validation
- Avoid breaking pruning guarantees
- Extend HAL instead of patching logic
- Remain testable without ROCm hardware

---

End of architecture overview.
