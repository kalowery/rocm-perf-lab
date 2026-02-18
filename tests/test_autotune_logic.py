import numpy as np
from rocm_perf_lab.analysis.pruning import prune_configs


def test_prune_configs_reduces_candidates():
    predictions = np.array([1.0, 2.0, 10.0, 0.9])
    best_runtime = 1.0
    prune_factor = 2.0

    candidates = prune_configs(predictions, best_runtime, prune_factor)

    assert 2 not in candidates  # 10.0 should be pruned
    assert 0 in candidates
    assert 1 in candidates
    assert 3 in candidates
