from rocm_perf_lab.analysis.critical_path import analyze_critical_path
from .utils import create_test_db


def test_symbol_aggregation():
    db = create_test_db(
        dispatch_rows=[
            (1, 1, 1, 0, 10),
            (2, 2, 1, 10, 30),
            (3, 1, 1, 30, 45),  # same symbol as dispatch 1
        ],
        symbol_rows=[
            (1, "A", "A"),
            (2, "B", "B"),
        ],
    )

    res = analyze_critical_path(db)

    # A total duration = 10 + 15 = 25
    # B duration = 20
    assert res.dominant_symbol_name == "A"
