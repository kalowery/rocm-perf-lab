import re
from pathlib import Path


UNSAFE_PATTERNS = [
    "__syncthreads",
    "atomic",
    "break",
    "return",
    "goto",
]


def is_loop_safe(loop_body: str) -> bool:
    return not any(p in loop_body for p in UNSAFE_PATTERNS)


def choose_unroll_factor(stall_fraction: float) -> int:
    factor = int(4 + 4 * stall_fraction)
    return max(2, min(8, factor))


def apply_loop_unroll(
    source_path: Path,
    stall_fraction: float,
    kernel_name: str,
) -> tuple[str, int]:

    src = source_path.read_text()

    kernel_pattern = rf"__global__\s+void\s+{kernel_name}\s*\("
    match = re.search(kernel_pattern, src)

    if not match:
        raise RuntimeError("Kernel not found in source")

    factor = choose_unroll_factor(stall_fraction)

    for_pattern = r"(for\s*\(.*?\)\s*\{)"
    loop_match = re.search(for_pattern, src[match.end():], re.DOTALL)

    if not loop_match:
        raise RuntimeError("No for-loop found for unroll")

    loop_start = match.end() + loop_match.start()

    brace_count = 0
    body_start = src.find("{", loop_start)
    i = body_start

    body_end = None
    while i < len(src):
        if src[i] == "{":
            brace_count += 1
        elif src[i] == "}":
            brace_count -= 1
            if brace_count == 0:
                body_end = i
                break
        i += 1

    if body_end is None:
        raise RuntimeError("Could not determine loop body")

    loop_body = src[body_start:body_end]

    if not is_loop_safe(loop_body):
        raise RuntimeError("Loop deemed unsafe for unroll")

    pragma = f"#pragma unroll {factor}\n"
    modified_src = src[:loop_start] + pragma + src[loop_start:]

    return modified_src, factor
