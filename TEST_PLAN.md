# Testing Plan for rocm-perf-lab

## Goals
- Maintain confidence in the deterministic replay/autotuning workflow described throughout the repository, with a particular focus on accuracy-critical components that interpret profiling data (`rocm_perf_lab/profiler/pipeline.py:1`, `rocm_perf_lab/analysis/critical_path.py:1`, `rocm_perf_lab/analysis/att_analysis.py:1`).
- Ensure CLI, optimizer and autotuner entry points (`rocm_perf_lab/cli/main.py:1`, `rocm_perf_lab/autotune/tuner.py:1`, `rocm_perf_lab/optimization/transform_loop_unroll.py:1`) behave predictably under both happy and failure paths without needing full hardware execution.
- Keep the integration tests that require ROCm + CDNA hardware (`tests/integration/test_minimal_replay.py:1`, `tests/integration/test_pointer_chase_replay.py:1`, see the documented setup in `README.md:413`) runnable and informative for regression tracking.

## Current Coverage Snapshot
- Unit tests currently exist for CLI version flag, CV classification, schema validation, pruning logic, and critical-path fragments (`tests/test_cli.py:1`, `tests/test_profile_logic.py:1`, `tests/test_schema.py:1`, `tests/test_autotune_logic.py:1`, `tests/critical_path/test_serial.py:1`).
- No direct coverage yet for the bulk of the numeric analysis tooling (`analysis/bottleneck_classifier`, `analysis/feature_engineering`, `analysis/regression`), autotune orchestration, LLM/loop-unroll helpers, HAL factory, or profiler helpers such as `att_runner`/`rocpd_detector`.
- Integration suite exercises isolation + replay, but only on CI with GPUs (`tests/integration/test_minimal_replay.py:1`, `tests/integration/test_pointer_chase_replay.py:1`).

## Test Matrix

### 1. Unit and infrastructure tests (mocked)
1. **CLI surface area** – Extend CI-grade coverage beyond `tests/test_cli.py:1` to guard `profile`, `autotune`, `optimize`, and `replay` subcommands. Use `typer.testing.CliRunner` with `monkeypatch` to stub out heavy helpers such as `build_profile`, `run_att`, `autotune`, and the binaries under `replay`. Validate JSON/quiet output formatting, warning paths (missing `rocpd` DB, signature errors) and exit codes.
2. **Profiler pipeline + schema** – Write targeted tests that inject fake `run_command` responses and ROI metadata to ensure:
   - `classify_cv` branches correctly (`tests/test_profile_logic.py:1`).
   - roofline fallback branches (missing metrics vs gfx942 path) and `persist_rocpd` picks up the right runtime from `sqlite` artifacts; use temp directories to write a fake `_results.db`.
   - `ProfileModel` validation still catches missing fields (`tests/test_schema.py:1`).
3. **HAL & resource modeling** – Verify `build_arch_from_agent_metadata` selects `CDNA3`, `CDNA2`, `RDNA2` or raises for unknown arch names (`rocm_perf_lab/hal/factory.py:1`, `rocm_perf_lab/hal/cdna3.py:1`). Mock metadata to confirm `compute_occupancy` saturates at hardware limits and gracefully handles zero divisions.
4. **Analysis primitives** – Add suites for:
   - `att_analysis` classification, empty results, and average-latency math with controlled `code.json` fixtures (`rocm_perf_lab/analysis/att_analysis.py:1`).
   - `bottleneck_classifier` thresholds (`rocm_perf_lab/analysis/bottleneck_classifier.py:1`).
   - `FeatureVectorizer.transform` handling missing keys and consistent ordering (`rocm_perf_lab/analysis/feature_engineering.py:1`).
   - `PerformanceRegressor` `fit`/`predict`/`metrics` guards and regression detection logic against synthetic dictionaries (`rocm_perf_lab/analysis/regression.py:1`).
   - `prune_configs` logic already touched but extend to `np.array` vs list inputs (`rocm_perf_lab/analysis/pruning.py:1`).
5. **Autotuner + optimization helpers** – Exercises for:
   - `build_static_features`, seed sampling and warning path when the regressor reports low confidence; stub `build_profile`/`FeatureVectorizer`/`PerformanceRegressor`/`prune_configs` as pure functions to inspect the returned `result` dict structure (`rocm_perf_lab/autotune/tuner.py:1`).
   - `apply_loop_unroll` safe-loop detection, pragma insertion, `choose_unroll_factor`, and failure modes (missing loop/kernel, unsafe patterns) using small synthetic CUDA snippets (`rocm_perf_lab/optimization/transform_loop_unroll.py:1`).
   - Variant directory naming (`create_variant_dir`, `save_variant_source`) for multiple proposals (`rocm_perf_lab/optimization/variant_manager.py:1`).
6. **LLM agent scaffolding** – Instead of running HIP/LLM, unit test string manipulation:
   - `replace_dominant_kernel` for straightforward replacements and for missing kernels (`rocm_perf_lab/llm/agent_loop.py:1`).
   - The signature enforcement branch by feeding a fake candidate kernel and raising before any subprocess invocation (mock `hipcc`).
   - Loop that terminates early when `min_improvement` trips or when `detect_regression` rejects, by mocking `build_profile`, `build_extended_profile`, `run_att`, and `detect_regression`.
7. **Profiler helpers** – Create tests for `_detect_latest_dispatch_dir`, `run_att`, and `detect_latest_rocpd_db` by generating temporary directories/files and monkeypatching `subprocess.run` so no real ROCm tooling is required (`rocm_perf_lab/profiler/att_runner.py:1`, `rocm_perf_lab/profiler/rocpd_detector.py:1`).

### 2. Component integration (hardware-agnostic but multi-module)
1. **Extended profile composition** – Feed a stubbed `base_profile` and synthetic `AttAnalysisResult`/`CriticalPathResult` objects to `build_extended_profile` to exercise the interaction between critical path, ATT, bottleneck classification, and headroom estimation (`rocm_perf_lab/profiler/extended_pipeline.py:1`). Confirm optional branches (missing rocpd, missing ATT) behave gracefully and that bottleneck reasoning surfaces in the result.
2. **Optimization entry points** – For both `autotune` and `optimize` CLI command handlers, use `CliRunner` + `monkeypatch` to simulate success/failure of their constituent helpers (profiling, ATT, loop-unroll). Assert the commands echo the expected messages and bail out early when heuristics deem optimization not worthwhile (`rocm_perf_lab/cli/main.py:1`).
3. **Critical-path variant coverage** – Existing tests touch serial, parallel, cross-stream flows (`tests/critical_path/test_serial.py:1`, `tests/critical_path/test_parallel.py:1`, `tests/critical_path/test_cross_stream.py:1`), but add a regression test for an empty DB and another that simulates the legacy `rocpd_kernel_dispatch_` table merging to keep `_get_table` covered.

### 3. GPU-backed integration/regression tests
1. **Fresh isolation + replay** – Keep running `tests/integration/test_minimal_replay.py:1` and `tests/integration/test_pointer_chase_replay.py:1` on MI300/MI325 hardware per the README workflow (`README.md:413`). Document any missing requirements (HIP SDK, `hipcc`, `rocprofv3`, isolate/replay builds) so CI can gate this suite appropriately.
2. **JSON replay output** – Add a variant of the full VM test that requests `--json` and ensures the JSON schema includes keys such as `execution.iterations` and `timing.average`. This helps guard the structured mode described near `docs/REPLAY.md` (refer to `README` sections that explain JSON output if needed).
3. **Regression tracking harness** – Capture replay timing artifacts into a dedicated log and compare against stored thresholds (e.g., guard against `api` changes by checking `timing.unit` etc.).

## Execution & Automation
- **Environment setup**: Follow `README.md:413` steps (`pip install -e .`, CMake builds of `rocm_perf_lab/isolate/tool` and `rocm_perf_lab/replay`). Document this once in `TEST_PLAN.md` so future engineers know the prerequisites.
- **Unit runs**: Run `pytest tests --maxfail=1 --durations=10` inside a ROCm-capable container but use fixtures/monkeypatching so they can also succeed on workstations without GPUs. Parameterize slow tests via `pytest.mark.skipif(not has_hipcc())` if necessary.
- **Integration runs**: Gate `pytest tests/integration -k minimal --runslow` behind an environment variable (e.g., `ROCM_HW_TESTS=1`) so CI can skip them on non-GPU runners while still allowing `mi300x`/`mi325` lab machines to run them routinely.
- **Data cleanup**: Have each GPU test consume `tmp_path`/`tmpdir` for binaries, clean `.rocpd_profile`, `.optimization`, and `isolate_capture_*` after the test (existing tests already rely on current working dir, but reinforce via fixtures).

## Observability & Follow-ups
- Track regression signals from `tests/integration` by exporting GPU runtime results (the script already prints `Average GPU time`, `Iterations`). Record them in a lightweight CSV under `testbed` for manual comparison until automation is in place.
- Add targeted coverage metrics (e.g., branch coverage for critical path, `att_analysis`) to the `pytest` job so we can spot regressions quickly.

## Next Steps
1. Translate each proposed unit/component test into a new file under `tests/` (e.g., `tests/test_profiler_pipeline.py`, `tests/test_llm_agent.py`, `tests/test_unroll.py`).
2. Implement the integration gating logic so `pytest tests/integration` is only invoked on GPU lab machines to avoid CI failures.
3. Document any new mocking/fixture helpers (e.g., fake `run_command` outputs) in `tests/README.md` or `docs/DEVELOPER_GUIDE.md` once the test scaffolding lands.
