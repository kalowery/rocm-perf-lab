from rocm_perf_lab.profiler.pipeline import classify_cv


def test_classify_cv_boundaries():
    assert classify_cv(0.05) == "stable"
    assert classify_cv(0.1) == "moderate"
    assert classify_cv(0.5001) == "unstable"
