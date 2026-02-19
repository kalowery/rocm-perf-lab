# CLI Reference — rocm-perf-lab (v2)

## profile

    rocm-perf profile <binary>

Advanced flags:
- --deep-analysis → Enable ATT + extended metrics
- --roofline → Enable FLOP/byte modeling
- --focus-critical → Restrict analysis to dominant kernel
- --json → Structured output
- --runs <int> → Number of measurement runs
- --debug → Show rocprof logs

Example:

    rocm-perf profile --deep-analysis --roofline --focus-critical "./app 100"

---

## llm-optimize

    rocm-perf llm-optimize <source.cu> "<binary>" [--auto-approve]

Behavior:
1. Profile baseline
2. Identify dominant kernel
3. Generate LLM patch
4. Compile variant
5. Profile full application
6. Apply regression detection
7. Detect dominance shift
8. Iterate

---

## prompt

    rocm-perf prompt <source.cu> "<binary>"

Generates structured hardware-grounded optimization prompt.

---

## autotune

Regression-based parameter search:

    rocm-perf autotune --space space.json --cmd-template "./kernel ..."

Separate from LLM structural optimization.
