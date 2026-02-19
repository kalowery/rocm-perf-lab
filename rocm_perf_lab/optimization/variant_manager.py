from pathlib import Path


def create_variant_dir(base_dir: Path, proposal_id: int) -> Path:
    opt_dir = base_dir / ".optimization"
    opt_dir.mkdir(exist_ok=True)

    proposal_dir = opt_dir / f"proposal_{proposal_id}"
    proposal_dir.mkdir(exist_ok=True)

    return proposal_dir


def save_variant_source(
    proposal_dir: Path,
    modified_source: str,
    original_path: Path,
) -> Path:

    version = proposal_dir.name.split("_")[-1]
    new_name = original_path.stem + f"_opt_v{version}.cu"

    out_path = proposal_dir / new_name
    out_path.write_text(modified_source)

    return out_path
