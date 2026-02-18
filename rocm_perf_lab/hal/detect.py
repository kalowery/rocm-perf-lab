import subprocess
import re
from .registry import create_architecture


def detect_architecture():
    try:
        result = subprocess.run(
            ["rocminfo"],
            capture_output=True,
            text=True,
            check=True
        )
    except Exception:
        raise RuntimeError("rocminfo not available.")

    output = result.stdout

    gfx_match = re.search(r"Name:\s+(gfx[0-9a-z]+)", output)
    cu_match = re.search(r"Compute Unit:\s+(\d+)", output)

    if not gfx_match or not cu_match:
        raise RuntimeError("Failed to parse rocminfo output.")

    gfx = gfx_match.group(1)
    compute_units = int(cu_match.group(1))

    return create_architecture(gfx, compute_units)
