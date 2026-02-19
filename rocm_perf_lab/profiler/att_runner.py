import subprocess
from pathlib import Path
from typing import Optional


def _detect_latest_dispatch_dir(base_dir: Path) -> Optional[Path]:
    candidates = []
    for p in base_dir.iterdir():
        if p.is_dir() and p.name.startswith("ui_output_agent_"):
            candidates.append(p)

    if not candidates:
        return None

    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


def run_att(cmd: str, workdir: Optional[Path] = None) -> Path:
    """
    Run rocprofv3 ATT pass for the given command.
    Returns path to latest ui_output_agent_* dispatch directory.
    """

    if workdir is None:
        workdir = Path.cwd()

    att_cmd = ["rocprofv3", "--att", "--", cmd]

    try:
        subprocess.run(att_cmd, cwd=workdir, check=True)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(
            "rocprofv3 ATT execution failed. "
            "If you see an error about 'rocprof-trace-decoder library path not found', "
            "ensure your ROCm installation is complete or that LD_LIBRARY_PATH includes "
            "the rocprof trace decoder library."
        ) from e

    dispatch_dir = _detect_latest_dispatch_dir(workdir)

    if dispatch_dir is None:
        raise RuntimeError(
            "ATT dispatch directory not found after rocprofv3 --att run. "
            "Ensure rocprofv3 completed successfully."
        )

    return dispatch_dir
