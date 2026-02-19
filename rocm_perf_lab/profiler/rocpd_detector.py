from pathlib import Path
from typing import Optional


def _detect_latest_hpcfund_dir(base_dir: Path) -> Optional[Path]:
    candidates = []
    for p in base_dir.iterdir():
        if p.is_dir() and p.name.endswith(".hpcfund"):
            candidates.append(p)

    if not candidates:
        return None

    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


def detect_latest_rocpd_db(base_dir: Optional[Path] = None) -> Optional[Path]:
    """
    Detect latest rocprof rocpd SQLite database file (*_results.db)
    inside newest *.hpcfund directory.
    """

    if base_dir is None:
        base_dir = Path.cwd()

    hpcfund_dir = _detect_latest_hpcfund_dir(base_dir)

    if hpcfund_dir is None:
        return None

    db_candidates = list(hpcfund_dir.glob("*_results.db"))

    if not db_candidates:
        return None

    db_candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return db_candidates[0]
