from pathlib import Path
from rocm_perf_lab.optimization.variant_manager import create_variant_dir, save_variant_source


def test_variant_dir_naming(tmp_path: Path):
    base = tmp_path / "project"
    base.mkdir()

    proposal1 = create_variant_dir(base, 1)
    proposal2 = create_variant_dir(base, 2)

    assert proposal1.exists()
    assert proposal2.exists()
    assert proposal1.name == "proposal_1"
    assert proposal2.name == "proposal_2"


def test_save_variant_source(tmp_path: Path):
    base = tmp_path / "project"
    base.mkdir()
    proposal = create_variant_dir(base, 3)

    original = base / "my_kernel.cu"
    original.write_text("kernel content")

    out_path = save_variant_source(proposal, "modified", original)

    assert out_path.exists()
    assert out_path.name == "my_kernel_opt_v3.cu"
    assert out_path.read_text() == "modified"
