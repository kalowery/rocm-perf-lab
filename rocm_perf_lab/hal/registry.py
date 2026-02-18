from .rdna2 import RDNA2
from .cdna2 import CDNA2


def create_architecture(gfx: str, compute_units: int):
    if gfx.startswith("gfx10"):
        return RDNA2(compute_units)
    if gfx == "gfx90a":
        return CDNA2(compute_units)
    raise ValueError(f"Unsupported architecture: {gfx}")
