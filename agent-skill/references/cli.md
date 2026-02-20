# CLI Reference — rocm-perf

Executable:

    rocm-perf

---

# profile

Profile a ROCm binary or kernel execution.

    rocm-perf profile <cmd>

Options:

- `--runs <int>`                Number of measurement runs (default: 3)
- `--rocprof / --no-rocprof`    Enable or disable rocprof tracing
- `--clock-mhz <float>`         Override GPU clock for peak FLOPs estimation
- `--quiet`                     Output runtime only
- `--debug`                     Show rocprof logs
- `--roofline`                  Enable roofline analysis
- `--focus-critical`            Enable critical path analysis (requires rocpd DB)
- `--deep-analysis`             Enable ATT deep analysis
- `--memory-bandwidth-gbps`     Override peak memory bandwidth
- `--json`                      Emit structured JSON output

Example:

    rocm-perf profile --roofline --focus-critical "./app 100"

---

# optimize

Deterministic loop-unroll optimization (non-LLM).

    rocm-perf optimize <source.cu> "<binary>"

Arguments:

- `source`   Standalone HIP kernel source file
- `binary`   Command used to execute the compiled program

Behavior:

1. Run extended profiling (roofline + critical path + ATT)
2. Identify dominant kernel on critical path
3. Verify bottleneck suitability (Memory Latency Bound)
4. Apply loop unroll transformation
5. Compile with `hipcc`
6. Re-profile
7. Accept if runtime improves (>2%)
8. Prompt user for confirmation before finalizing

---

# prompt

Generate an LLM optimization prompt from profiling data.

    rocm-perf prompt <source.cu> "<binary>"

Options:

- `--full-source`   Include full source file instead of dominant kernel only
- `--json`          Emit structured optimization context as JSON
- `--compact`       Emit compact LLM prompt
- `--runs <int>`    Number of measurement runs (default: 3)

This command does not execute an LLM; it prepares the optimization context.

---

# llm-optimize

Closed-loop LLM optimization using OpenAI.

    rocm-perf llm-optimize <source.cu> "<binary>"

Options:

- `--model <name>`            OpenAI model name (default: gpt-4.1)
- `--temperature <float>`     Sampling temperature (default: 0.2)
- `--max-iters <int>`         Maximum optimization iterations (default: 3)
- `--min-improvement <float>` Minimum fractional improvement required (default: 0.02)
- `--auto-approve`            Automatically apply accepted changes without prompt

Requires `OPENAI_API_KEY` environment variable.

Behavior:

1. Profile baseline
2. Generate LLM patch proposal
3. Compile variant
4. Re-profile
5. Accept only if measured improvement ≥ min-improvement
6. Iterate up to max-iters

---

# autotune

Adaptive regression-based parameter search.

    rocm-perf autotune \
        --space <space.json> \
        --cmd-template "<command>"

Options:

- `--seed <int>`              Random seed (default: 0)
- `--seed-fraction <float>`   Fraction of space for seed phase (default: 0.2)
- `--prune-factor <float>`    Pruning threshold factor (default: 1.75)
- `--json`                    Emit structured JSON output

Used for parameter tuning, separate from structural kernel transformations.
