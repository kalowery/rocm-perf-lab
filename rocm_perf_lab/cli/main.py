import typer
import json
from rocm_perf_lab.profiler.pipeline import build_profile
from rocm_perf_lab.profiler.extended_pipeline import build_extended_profile
from rocm_perf_lab.profiler.att_runner import run_att
from rocm_perf_lab.profiler.rocpd_detector import detect_latest_rocpd_db
from rocm_perf_lab.autotune.tuner import autotune as run_autotune

app = typer.Typer(no_args_is_help=True)

# ==========================================================
# Replay & VM Diagnostics Commands (registered early)
# ==========================================================

replay_app = typer.Typer(help="Replay and VM diagnostics tools.")

@replay_app.command("full-vm")
def replay_full_vm(
    capture_dir: str = typer.Option(..., "--capture-dir", help="Path to isolate capture directory.")
):
    """
    Perform full VM-faithful replay using reconstructed device memory.
    """
    from pathlib import Path
    import subprocess

    binary = Path(__file__).resolve().parent.parent / "replay" / "build" / "rocm_perf_replay_full_vm"

    if not binary.exists():
        typer.echo("Replay binary not built. Run CMake build in rocm_perf_lab/replay.")
        raise typer.Exit(code=1)

    capture_path = Path(capture_dir).resolve()
    result = subprocess.run([str(binary), str(capture_path)])
    raise typer.Exit(code=result.returncode)


@replay_app.command("reserve-check")
def replay_reserve_check(
    capture_dir: str = typer.Option(..., "--capture-dir", help="Path to isolate capture directory.")
):
    """
    Validate fixed-address VM reservation feasibility.
    """
    from pathlib import Path
    import subprocess

    binary = Path(__file__).resolve().parent.parent / "replay" / "build" / "vm_reserve_only"

    if not binary.exists():
        typer.echo("Diagnostic binary not built. Run CMake build in rocm_perf_lab/replay.")
        raise typer.Exit(code=1)

    capture_path = Path(capture_dir).resolve()
    result = subprocess.run([str(binary), str(capture_path)])
    raise typer.Exit(code=result.returncode)


app.add_typer(replay_app, name="replay")


@app.callback()
def main(
    version: bool = typer.Option(
        False,
        "--version",
        help="Show version and exit.",
        is_eager=True,
    )
):
    if version:
        from rocm_perf_lab import __version__
        typer.echo(__version__)
        raise typer.Exit()


@app.command()
def profile(
    cmd: str,
    runs: int = typer.Option(3, "--runs", help="Number of measurement runs."),
    rocprof: bool = typer.Option(True, "--rocprof/--no-rocprof", help="Enable or disable rocprof tracing."),
    clock_mhz: float = typer.Option(None, "--clock-mhz", help="Override GPU clock for peak FLOPs estimation."),
    quiet: bool = typer.Option(False, "--quiet", help="Suppress non-essential output."),
    debug: bool = typer.Option(False, "--debug", help="Enable debug output (shows rocprof logs)."),
    roofline: bool = typer.Option(False, "--roofline", help="Enable roofline analysis using hardware counters."),
    focus_critical: bool = typer.Option(False, "--focus-critical", help="Enable critical path analysis (requires rocpd DB)."),
    deep_analysis: bool = typer.Option(False, "--deep-analysis", help="Enable ATT deep microarchitectural analysis."),
    memory_bandwidth_gbps: float = typer.Option(None, "--memory-bandwidth-gbps", help="Override peak memory bandwidth in GB/s."),
    json_output: bool = typer.Option(False, "--json", help="Emit structured JSON output.")
):
    """Profile a ROCm kernel or binary."""

    if focus_critical or deep_analysis:
        # First run base profile to generate rocpd DB
        base_profile = build_profile(
            cmd=cmd,
            runs=runs,
            use_rocprof=rocprof,
            clock_mhz=clock_mhz,
            debug=debug,
            roofline=roofline,
            memory_bandwidth_gbps=memory_bandwidth_gbps,
            persist_rocpd=focus_critical,
        )

        rocpd_db_path = None
        att_dispatch_dir = None

        if focus_critical:
            # If we persisted rocpd output, prefer DB inside .rocpd_profile
            if focus_critical:
                from pathlib import Path
                import glob

                profile_dir = Path(".rocpd_profile")
                if profile_dir.exists():
                    db_files = glob.glob(str(profile_dir / "**/*_results.db"), recursive=True)
                    rocpd_db_path = Path(max(db_files, key=lambda p: Path(p).stat().st_mtime)) if db_files else None
                else:
                    rocpd_db_path = detect_latest_rocpd_db()
            else:
                rocpd_db_path = detect_latest_rocpd_db()

            if rocpd_db_path is None:
                typer.echo("Warning: rocpd database not found. Critical path analysis skipped.")

        if deep_analysis:
            typer.echo("Running ATT deep analysis (this may take time)...")
            att_dispatch_dir = run_att(cmd)

        result = build_extended_profile(
            base_profile=base_profile,
            rocpd_db_path=rocpd_db_path,
            att_dispatch_dir=att_dispatch_dir,
        )
    else:
        result = build_profile(
            cmd=cmd,
            runs=runs,
            use_rocprof=rocprof,
            clock_mhz=clock_mhz,
            debug=debug,
            roofline=roofline,
            memory_bandwidth_gbps=memory_bandwidth_gbps,
        )

    if json_output:
        typer.echo(json.dumps(result, indent=2))
        return

    if quiet:
        typer.echo(f"{result['runtime_ms']:.6f}")
        return

    typer.echo(f"Runtime: {result['runtime_ms']:.3f} ms")
    typer.echo(f"CV: {result['stability']['cv']:.4f}")

    if result["stability"]["classification"] == "unstable":
        typer.echo("WARNING: Run variability is high (unstable measurements).")

    if result["occupancy"]:
        typer.echo(f"Occupancy: {result['occupancy']['theoretical']:.2f}")


@app.command(name="autotune")
def autotune(
    space: str = typer.Option(..., "--space", help="Path to JSON file containing expanded search space."),
    cmd_template: str = typer.Option(..., "--cmd-template", help="Command template with placeholders for parameters."),
    seed: int = typer.Option(0, "--seed", help="Random seed for seed-phase sampling."),
    seed_fraction: float = typer.Option(0.2, "--seed-fraction", help="Fraction of search space to use for seed phase."),
    prune_factor: float = typer.Option(1.75, "--prune-factor", help="Pruning threshold factor relative to current best runtime."),
    json_output: bool = typer.Option(False, "--json", help="Emit structured JSON output.")
):
    """Adaptive regression-based autotuning for ROCm kernels."""

    with open(space) as f:
        search_space = json.load(f)

    result = run_autotune(
        search_space=search_space,
        cmd_template=cmd_template,
        seed=seed,
        seed_fraction=seed_fraction,
        prune_factor=prune_factor,
    )

    if json_output:
        typer.echo(json.dumps(result, indent=2))
        return

    typer.echo(f"Best runtime: {result['best_config']['runtime_ms']:.3f} ms")

    if "warning" in result:
        typer.echo("WARNING: Model confidence is low (R² < 0.75). Pruning may be unreliable.")




@app.command()
def optimize(
    source: str,
    binary: str,
    runs: int = 3,
):
    """
    Optimize a standalone HIP kernel using extended profiling + loop unroll.
    """
    from pathlib import Path
    import subprocess
    import shutil

    from rocm_perf_lab.analysis.optimization_score import compute_optimization_score
    from rocm_perf_lab.optimization.transform_loop_unroll import apply_loop_unroll
    from rocm_perf_lab.optimization.variant_manager import create_variant_dir, save_variant_source

    source_path = Path(source)
    binary_cmd = binary

    if not source_path.exists():
        typer.echo("Source file not found.")
        raise typer.Exit(code=1)

    typer.echo("=== Baseline Extended Profiling ===")

    # Run base profile first (persist rocpd for critical-path)
    base_profile = build_profile(
        cmd=binary_cmd,
        runs=runs,
        use_rocprof=True,
        roofline=True,
        persist_rocpd=True,
    )

    # Detect rocpd DB inside .rocpd_profile
    from pathlib import Path
    import glob

    profile_dir = Path(".rocpd_profile")
    db_files = glob.glob(str(profile_dir / "**/*_results.db"), recursive=True)
    rocpd_db_path = Path(max(db_files, key=lambda p: Path(p).stat().st_mtime)) if db_files else None

    # Run ATT
    att_dispatch_dir = run_att(binary_cmd)

    baseline = build_extended_profile(
        base_profile=base_profile,
        rocpd_db_path=rocpd_db_path,
        att_dispatch_dir=att_dispatch_dir,
    )

    headroom = baseline.get("headroom_fraction", 0.0)
    critical = baseline.get("critical_path", {})
    dominant_symbol = critical.get("dominant_symbol")
    critical_fraction = critical.get("fraction", 0.0)
    bottleneck = baseline.get("bottleneck", {}).get("primary")
    stall_fraction = baseline.get("att", {}).get("stall_fraction", 0.0)
    runtime_baseline = baseline.get("runtime_ms", 0.0)

    if dominant_symbol is None:
        typer.echo("No dominant symbol identified. Aborting optimization.")
        raise typer.Exit()

    if headroom < 0.10 or critical_fraction < 0.20:
        typer.echo("Optimization not worthwhile based on headroom/critical path.")
        raise typer.Exit()

    if bottleneck != "Memory Latency Bound":
        typer.echo(f"Bottleneck is {bottleneck}. Loop unroll not applicable.")
        raise typer.Exit()

    score = compute_optimization_score(headroom, critical_fraction)

    if score.priority_score < 0.05:
        typer.echo("Priority score too low. Skipping optimization.")
        raise typer.Exit()

    typer.echo("=== Applying Loop Unroll Transformation ===")

    try:
        modified_src, factor = apply_loop_unroll(
            source_path,
            stall_fraction,
            dominant_symbol,
        )
    except Exception as e:
        typer.echo(f"Transformation failed: {e}")
        raise typer.Exit()

    proposal_dir = create_variant_dir(source_path.parent, 1)
    variant_source_path = save_variant_source(
        proposal_dir,
        modified_src,
        source_path,
    )

    variant_binary = proposal_dir / "variant_binary"

    typer.echo("=== Compiling Variant ===")

    subprocess.run([
        "hipcc",
        "-O3",
        str(variant_source_path),
        "-o",
        str(variant_binary),
    ], check=True)

    typer.echo("=== Re-Profiling Variant ===")

    # Re-profile variant (persist rocpd)
    base_profile_new = build_profile(
        cmd=str(variant_binary),
        runs=runs,
        use_rocprof=True,
        roofline=True,
        persist_rocpd=True,
    )

    profile_dir_new = Path(".rocpd_profile")
    db_files_new = glob.glob(str(profile_dir_new / "**/*_results.db"), recursive=True)
    rocpd_db_path_new = Path(max(db_files_new, key=lambda p: Path(p).stat().st_mtime)) if db_files_new else None

    att_dispatch_dir_new = run_att(str(variant_binary))

    new_profile = build_extended_profile(
        base_profile=base_profile_new,
        rocpd_db_path=rocpd_db_path_new,
        att_dispatch_dir=att_dispatch_dir_new,
    )

    runtime_new = new_profile.get("runtime_ms", 0.0)

    if runtime_new <= 0:
        typer.echo("Invalid runtime for variant.")
        raise typer.Exit()

    speedup = runtime_baseline / runtime_new

    typer.echo("\n=== Optimization Proposal ===")
    typer.echo(f"Target Kernel: {dominant_symbol}")
    typer.echo(f"Transformation: Loop Unroll (factor={factor})")
    typer.echo(f"Baseline Runtime: {runtime_baseline:.6f} ms")
    typer.echo(f"New Runtime:      {runtime_new:.6f} ms")
    typer.echo(f"Speedup:          {speedup:.3f}x")

    if speedup < 1.02:
        typer.echo("Improvement < 2%. Rejecting.")
        raise typer.Exit()

    response = input("Apply change? [y/n]: ").strip().lower()

    if response != "y":
        typer.echo("Change rejected.")
        raise typer.Exit()

    final_path = source_path.parent / (source_path.stem + "_opt_v1.cu")
    shutil.copyfile(variant_source_path, final_path)

    typer.echo(f"Optimized file written to {final_path}")




@app.command()
def prompt(
    source: str,
    binary: str,
    full_source: bool = typer.Option(False, "--full-source", help="Include full source file instead of dominant kernel only."),
    json_output: bool = typer.Option(False, "--json", help="Emit structured optimization context as JSON."),
    compact: bool = typer.Option(False, "--compact", help="Emit compact LLM prompt."),
    runs: int = 3,
):
    """
    Generate an LLM optimization prompt from profiling data.
    """
    from pathlib import Path
    import glob
    import json

    from rocm_perf_lab.llm.prompt_builder import (
        build_optimization_context,
        build_llm_prompt,
    )

    source_path = Path(source)
    if not source_path.exists():
        typer.echo("Source file not found.")
        raise typer.Exit(code=1)

    typer.echo("=== Running Extended Profiling ===")

    # Base profile with persistent rocpd
    base_profile = build_profile(
        cmd=binary,
        runs=runs,
        use_rocprof=True,
        roofline=True,
        persist_rocpd=True,
    )

    profile_dir = Path(".rocpd_profile")
    db_files = glob.glob(str(profile_dir / "**/*_results.db"), recursive=True)
    rocpd_db_path = Path(max(db_files, key=lambda p: Path(p).stat().st_mtime)) if db_files else None

    att_dispatch_dir = run_att(binary)

    extended = build_extended_profile(
        base_profile=base_profile,
        rocpd_db_path=rocpd_db_path,
        att_dispatch_dir=att_dispatch_dir,
    )

    context = build_optimization_context(
        source_path=source_path,
        extended_profile=extended,
        full_source=full_source,
    )

    if json_output:
        typer.echo(json.dumps(context, indent=2))
        return

    prompt_text = build_llm_prompt(context, compact=compact)
    typer.echo(prompt_text)




@app.command()
def llm_optimize(
    source: str,
    binary: str,
    model: str = typer.Option("gpt-4.1", "--model", help="OpenAI model name."),
    temperature: float = typer.Option(0.2, "--temperature", help="Sampling temperature."),
    max_iters: int = typer.Option(3, "--max-iters", help="Maximum LLM optimization iterations."),
    min_improvement: float = typer.Option(0.02, "--min-improvement", help="Minimum fractional improvement required."),
    auto_approve: bool = typer.Option(False, "--auto-approve", help="Automatically continue without user confirmation."),
):
    """
    Run closed-loop LLM optimization using OpenAI.
    Requires OPENAI_API_KEY environment variable.
    """
    from pathlib import Path
    from rocm_perf_lab.llm.agent_loop import run_llm_optimization_loop
    from rocm_perf_lab.llm.providers.openai_provider import openai_llm_callable

    source_path = Path(source)
    if not source_path.exists():
        typer.echo("Source file not found.")
        raise typer.Exit(code=1)

    llm_callable = openai_llm_callable(model=model, temperature=temperature)

    run_llm_optimization_loop(
        source_path=source_path,
        binary_cmd=binary,
        llm_callable=llm_callable,
        max_iters=max_iters,
        min_improvement=min_improvement,
        auto_approve=auto_approve,
    )


if __name__ == "__main__":
    app()
