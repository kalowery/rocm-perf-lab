import pytest
from pathlib import Path

from rocm_perf_lab.optimization.transform_loop_unroll import apply_loop_unroll, choose_unroll_factor


def _write_kernel(tmp_path: Path, loop_body: str):
    src = f"""__global__ void my_kernel(int *data) {{\n    for (int i = 0; i < 4; ++i) {{{loop_body}}}\n}}"""
    path = tmp_path / "kernel.cu"
    path.write_text(src)
    return path


def test_apply_loop_unroll_inserts_pragma(tmp_path: Path):
    path = _write_kernel(tmp_path, "data[i] = i;\n        ")
    modified, factor = apply_loop_unroll(path, stall_fraction=0.5, kernel_name="my_kernel")

    assert f"#pragma unroll {factor}" in modified
    assert "data[i] = i;" in modified


def test_apply_loop_unroll_replaces_existing_pragma(tmp_path: Path):
    src = """__global__ void my_kernel(int *data) {\n#pragma unroll 2\n    for (int i = 0; i < 4; ++i) { data[i] = i; }\n}"""
    path = tmp_path / "kernel.cu"
    path.write_text(src)

    modified, factor = apply_loop_unroll(path, stall_fraction=0.8, kernel_name="my_kernel")

    assert f"#pragma unroll {factor}" in modified
    assert "#pragma unroll 2" not in modified


def test_apply_loop_unroll_missing_kernel(tmp_path: Path):
    path = tmp_path / "kernel.cu"
    path.write_text("__global__ void other_kernel() {}")

    with pytest.raises(RuntimeError):
        apply_loop_unroll(path, stall_fraction=0.1, kernel_name="missing_kernel")


def test_apply_loop_unroll_rejects_unsafe_loop(tmp_path: Path):
    path = _write_kernel(tmp_path, "atomicAdd(&data[i], 1);\n        ")

    with pytest.raises(RuntimeError):
        apply_loop_unroll(path, stall_fraction=0.3, kernel_name="my_kernel")


def test_choose_unroll_factor_bounds():
    assert 2 <= choose_unroll_factor(0.0) <= 8
    assert 2 <= choose_unroll_factor(1.0) <= 8
