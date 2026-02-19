from rocm_perf_lab.analysis.critical_path import analyze_critical_path
from .utils import create_test_db


def test_serial_chain():
    db = create_test_db(
        dispatch_rows=[
            (1, 1, 1, 0, 10),
            (2, 2, 1, 10, 30),
            (3, 3, 1, 30, 35),
        ],
        symbol_rows=[
            (1, "A", "A"),
            (2, "B", "B"),
            (3, "C", "C"),
        ],
    )

    res = analyze_critical_path(db)

    assert res.critical_kernel_ids == [1, 2, 3]
    assert res.dominant_symbol_name == "B"
