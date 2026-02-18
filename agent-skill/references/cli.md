# CLI Reference — rocm-perf-lab

## profile Command

```
rocm-perf profile <binary>
```

Options:

- `--runs <int>` — Number of measurement runs
- `--rocprof / --no-rocprof` — Enable or disable hardware tracing
- `--clock-mhz <float>` — Override clock for peak FLOPs estimation
- `--quiet` — Output runtime only
- `--debug` — Show rocprof logs
- `--json` — Structured JSON output

Behavior matrix:

| Mode | Output |
|------|--------|
| default | Human summary |
| --quiet | Single numeric runtime |
| --json | Clean structured JSON |
| --debug | Shows rocprof output |

---

## autotune Command

```
rocm-perf autotune \
  --space search_space.json \
  --cmd-template "./kernel --bm {BLOCK_M}"
```

Options:

- `--seed` — RNG seed
- `--seed-fraction` — Fraction used for seed phase
- `--prune-factor` — Pruning aggressiveness
- `--json` — Structured output

---

## version

```
rocm-perf --version
```

Returns tool version.
