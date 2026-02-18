from .rdna2 import RDNA2
from .cdna2 import CDNA2
from .cdna3 import CDNA3


def build_arch_from_agent_metadata(meta: dict):
    arch = meta.get("arch_name", "").lower()

    if arch == "gfx942":
        return CDNA3(**meta)

    if arch.startswith("gfx90"):
        return CDNA2(**meta)

    if arch.startswith("gfx103"):
        return RDNA2(**meta)

    raise ValueError(
        f"Unsupported architecture '{arch}'. Add a HAL implementation for this architecture."
    )
