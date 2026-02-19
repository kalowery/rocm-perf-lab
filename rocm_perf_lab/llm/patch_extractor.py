import re


def extract_cpp_patch(response: str, expected_kernel_name: str) -> str:
    """
    Extract first ```cpp fenced block and validate it contains
    the expected __global__ kernel.
    """
    fence_pattern = r"```cpp(.*?)```"
    match = re.search(fence_pattern, response, re.DOTALL)

    if not match:
        raise RuntimeError("No ```cpp fenced block found in LLM response.")

    code = match.group(1).strip()

    if "__global__" not in code:
        raise RuntimeError("Extracted patch does not contain a __global__ kernel.")

    kernel_base = expected_kernel_name.split("(")[0] if expected_kernel_name else None
    if kernel_base and kernel_base not in code:
        raise RuntimeError("Patched kernel does not match dominant kernel name.")

    return code
