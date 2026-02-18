import typer
import json
from rocm_perf_lab.profiler.pipeline import build_profile
from rocm_perf_lab.autotune.tuner import autotune as run_autotune

app = typer.Typer(no_args_is_help=True)


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
    memory_bandwidth_gbps: float = typer.Option(None, "--memory-bandwidth-gbps", help="Override peak memory bandwidth in GB/s."),
    json_output: bool = typer.Option(False, "--json", help="Emit structured JSON output.")
):
    """Profile a ROCm kernel or binary."""

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


if __name__ == "__main__":
    app()
