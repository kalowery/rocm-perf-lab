import subprocess
import sqlite3
import tempfile
import os
from dataclasses import dataclass


@dataclass
class RocprofResult:
    kernel_name: str
    kernel_time_ms: float
    vgpr_per_thread: int | None
    sgpr_per_wave: int | None
    lds_bytes: int | None
    grid: tuple[int, int, int] | None
    block: tuple[int, int, int] | None
    agent_metadata: dict | None = None


def run_with_rocprof(cmd: str, debug: bool = False) -> RocprofResult:
    with tempfile.TemporaryDirectory() as tmpdir:
        # Execute rocprofv3 with explicit output directory and rocpd format
        rocprof_cmd = [
            "rocprofv3",
            "--kernel-trace",
            "-d",
            tmpdir,
            "-f",
            "rocpd",
            "--",
        ] + cmd.split()

        try:
            env = os.environ.copy()
            env["HOME"] = tmpdir
            env["ROCPROFILER_HOME"] = tmpdir
            env["XDG_CACHE_HOME"] = tmpdir
            if debug:
                subprocess.run(rocprof_cmd, check=True, env=env)
            else:
                subprocess.run(
                    rocprof_cmd,
                    check=True,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    env=env,
                )
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"rocprofv3 execution failed: {e}")

        # Discover generated results.db recursively
        import glob
        db_files = glob.glob(os.path.join(tmpdir, "**/*_results.db"), recursive=True)
        if not db_files:
            raise RuntimeError("rocprofv3 did not produce results.db")

        return parse_rocpd_sqlite(db_files[0])


def demangle(name: str) -> str:
    try:
        result = subprocess.run(
            ["c++filt", name],
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except Exception:
        return name


def is_runtime_kernel(name: str) -> bool:
    return (
        name.startswith("__amd_rocclr_")
        or name.startswith("hip")
        or name.startswith("hsa_")
    )


def run_with_rocprof_counters(cmd: str, metrics: list[str], debug: bool = False):
    """Run rocprofv3 in counter mode and return metric dict."""
    with tempfile.TemporaryDirectory() as tmpdir:
        metric_arg = ",".join(metrics)
        rocprof_cmd = [
            "rocprofv3",
            "--kernel-trace",
            "--stats",
            "--pmc",
            metric_arg,
            "-d",
            tmpdir,
            "-f",
            "rocpd",
            "--",
        ] + cmd.split()

        try:
            env = os.environ.copy()
            env["HOME"] = tmpdir
            env["ROCPROFILER_HOME"] = tmpdir
            env["XDG_CACHE_HOME"] = tmpdir
            if debug:
                subprocess.run(rocprof_cmd, check=True, env=env)
            else:
                subprocess.run(
                    rocprof_cmd,
                    check=True,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    env=env,
                )
        except subprocess.CalledProcessError:
            return None

        import glob
        db_files = glob.glob(os.path.join(tmpdir, "**/*_results.db"), recursive=True)
        if not db_files:
            return None

        return parse_rocpd_metrics(db_files[0], metrics)


def parse_rocpd_metrics(db_path: str, metrics: list[str]):
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name LIKE 'rocpd_pmc_event%';")
    row = cur.fetchone()
    if not row:
        conn.close()
        return None

    pmc_table = row[0]

    metric_values = {}

    for metric in metrics:
        try:
            cur.execute(
                f"SELECT SUM(value) FROM {pmc_table} WHERE name = ?;",
                (metric,),
            )
            value = cur.fetchone()[0]
            metric_values[metric] = value or 0.0
        except Exception:
            metric_values[metric] = 0.0

    conn.close()
    return metric_values


def parse_rocpd_sqlite(db_path: str) -> RocprofResult:
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    # Find required tables dynamically
    cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name LIKE 'rocpd_kernel_dispatch%';")
    dispatch_table = cur.fetchone()[0]

    cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name LIKE 'rocpd_info_kernel_symbol%';")
    kernel_info_table = cur.fetchone()[0]

    # Get all dispatches ordered by duration DESC
    cur.execute(
        f"""
        SELECT kernel_id, agent_id, start, end,
               workgroup_size_x, workgroup_size_y, workgroup_size_z,
               grid_size_x, grid_size_y, grid_size_z,
               (end - start) AS duration
        FROM {dispatch_table}
        ORDER BY duration DESC;
        """
    )
    dispatch_rows = cur.fetchall()

    selected = None

    selected_agent_id = None

    for row in dispatch_rows:
        kernel_id, agent_id, start, end, wx, wy, wz, gx, gy, gz, duration_ns = row

        # Lookup kernel metadata
        cur.execute(
            f"SELECT kernel_name, display_name, arch_vgpr_count, sgpr_count, group_segment_size FROM {kernel_info_table} WHERE id = ?;",
            (kernel_id,),
        )
        meta = cur.fetchone()

        if not meta:
            continue

        mangled_name, display_name, arch_vgpr_count, sgpr_count, group_segment_size = meta

        name = display_name or (demangle(mangled_name) if mangled_name else "unknown")

        if not is_runtime_kernel(name):
            selected = (name, duration_ns, wx, wy, wz, gx, gy, gz,
                        arch_vgpr_count, sgpr_count, group_segment_size)
            selected_agent_id = agent_id
            break

    # Fallback to longest if all were runtime kernels
    if not selected and dispatch_rows:
        row = dispatch_rows[0]
        kernel_id, agent_id, start, end, wx, wy, wz, gx, gy, gz, duration_ns = row

        cur.execute(
            f"SELECT kernel_name, display_name, arch_vgpr_count, sgpr_count, group_segment_size FROM {kernel_info_table} WHERE id = ?;",
            (kernel_id,),
        )
        meta = cur.fetchone()

        if meta:
            mangled_name, display_name, arch_vgpr_count, sgpr_count, group_segment_size = meta
            name = display_name or (demangle(mangled_name) if mangled_name else "unknown")
            selected = (name, duration_ns, wx, wy, wz, gx, gy, gz,
                        arch_vgpr_count, sgpr_count, group_segment_size)
            selected_agent_id = agent_id

    if not selected:
        conn.close()
        raise RuntimeError("No valid kernel dispatch found")

    # Extract agent metadata (strict validation)
    if selected_agent_id is None:
        conn.close()
        raise RuntimeError("Agent ID not found for selected kernel dispatch")

    cur.execute(
        "SELECT name, extdata FROM rocpd_info_agent WHERE id = ?;",
        (selected_agent_id,),
    )
    agent_row = cur.fetchone()

    if not agent_row:
        conn.close()
        raise RuntimeError("Agent metadata not found in rocpd_info_agent")

    name, extdata = agent_row

    if not extdata:
        conn.close()
        raise RuntimeError("Agent extdata missing in rocpd_info_agent")

    import json
    try:
        agent_json = json.loads(extdata)
    except Exception as e:
        conn.close()
        raise RuntimeError(f"Failed to parse agent extdata JSON: {e}")

    required = [
        "cu_count",
        "simd_per_cu",
        "max_waves_per_cu",
        "wave_front_size",
        "max_engine_clk_fcompute",
    ]

    missing = [k for k in required if k not in agent_json]
    if missing:
        conn.close()
        raise RuntimeError(
            f"Incomplete hardware metadata in rocpd extdata; missing fields: {missing}"
        )

    agent_metadata = {
        "arch_name": (name or "").strip().lower(),
        "cu_count": agent_json["cu_count"],
        "simd_per_cu": agent_json["simd_per_cu"],
        "max_waves_per_cu": agent_json["max_waves_per_cu"],
        "wave_size": agent_json["wave_front_size"],
        "max_clock_mhz": agent_json["max_engine_clk_fcompute"],
    }

    conn.close()

    (kernel_name, duration_ns, wx, wy, wz, gx, gy, gz,
     vgpr, sgpr, lds) = selected

    duration_ms = duration_ns / 1e6

    return RocprofResult(
        kernel_name=kernel_name,
        kernel_time_ms=duration_ms,
        vgpr_per_thread=vgpr,
        sgpr_per_wave=sgpr,
        lds_bytes=lds,
        grid=(gx, gy, gz),
        block=(wx, wy, wz),
        agent_metadata=agent_metadata,
    )
