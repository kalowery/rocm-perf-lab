from rocm_perf_lab.analysis.critical_path import analyze_critical_path
from .utils import create_test_db


def test_parallel_independent():
    db = create_test_db(
        dispatch_rows=[
            (1, 1, 1, 0, 10),
            (2, 2, 2, 0, 30),
        ],
        symbol_rows=[
            (1, "A", "A"),
            (2, "B", "B"),
        ],
    )

    res = analyze_critical_path(db)

    assert res.critical_path_ns == 30
    assert res.dominant_symbol_name == "B"
