from rocm_perf_lab.profiler.pipeline import classify_cv


def test_classify_cv_stable():
    assert classify_cv(0.03) == "stable"


def test_classify_cv_moderate():
    assert classify_cv(0.08) == "moderate"


def test_classify_cv_unstable():
    assert classify_cv(0.2) == "unstable"
