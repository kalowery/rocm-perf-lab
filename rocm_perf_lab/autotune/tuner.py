import random
import numpy as np
from rocm_perf_lab.profiler.pipeline import build_profile
from rocm_perf_lab.analysis.regression import PerformanceRegressor
from rocm_perf_lab.analysis.feature_engineering import FeatureVectorizer
from rocm_perf_lab.analysis.pruning import prune_configs


# Static features only (available before profiling)
STATIC_FEATURE_ORDER = [
    "ACC",
    "ACC_sq",
    "BLOCK_K",
    "num_warps",
    "num_stages",
    "threads_per_block",
]


def build_static_features(config: dict):
    acc = config["BLOCK_M"] * config["BLOCK_N"]
    threads_per_block = config.get("threads_per_block", 256)

    return {
        "ACC": acc,
        "ACC_sq": acc * acc,
        "BLOCK_K": config["BLOCK_K"],
        "num_warps": config["num_warps"],
        "num_stages": config["num_stages"],
        "threads_per_block": threads_per_block,
    }


def autotune(
    search_space: list[dict],
    cmd_template: str,
    seed: int = 0,
    seed_fraction: float = 0.2,
    prune_factor: float = 1.75,
):
    random.seed(seed)

    total_configs = len(search_space)
    seed_size = max(10, int(seed_fraction * total_configs))
    seed_indices = random.sample(range(total_configs), min(seed_size, total_configs))

    seed_results = []
    feature_dicts = []
    runtimes = []

    # Seed phase (profile only seed configs)
    for idx in seed_indices:
        config = search_space[idx]
        cmd = cmd_template.format(**config)
        profile = build_profile(cmd, use_rocprof=True)

        runtime = profile["runtime_ms"]
        features = build_static_features(config)

        seed_results.append((idx, runtime))
        feature_dicts.append(features)
        runtimes.append(runtime)

    best_runtime = min(runtimes)

    vectorizer = FeatureVectorizer(STATIC_FEATURE_ORDER)
    X = vectorizer.transform(feature_dicts)
    y = np.array(runtimes)

    regressor = PerformanceRegressor(degree=2)
    regressor.fit(X, y)

    metrics = regressor.metrics()

    # Prediction phase (no profiling here)
    if metrics["r2"] < 0.75:
        candidate_indices = list(range(total_configs))
    else:
        all_feature_dicts = [build_static_features(cfg) for cfg in search_space]
        X_all = vectorizer.transform(all_feature_dicts)
        predictions = regressor.predict(X_all)
        candidate_indices = prune_configs(predictions, best_runtime, prune_factor)

    evaluated = set(seed_indices)
    final_results = seed_results.copy()

    # Confirm phase (profile only pruned candidates)
    for idx in candidate_indices:
        if idx in evaluated:
            continue
        config = search_space[idx]
        cmd = cmd_template.format(**config)
        profile = build_profile(cmd, use_rocprof=True)

        runtime = profile["runtime_ms"]
        final_results.append((idx, runtime))

    best_idx, best_runtime = min(final_results, key=lambda x: x[1])
    best_config = search_space[best_idx]

    result = {
        "search_space_size": total_configs,
        "evaluated_configs": len(final_results),
        "best_config": {
            "parameters": best_config,
            "runtime_ms": best_runtime,
        },
        "model": {
            "r2": metrics["r2"],
            "residual_std": metrics["residual_std"],
        }
    }

    if metrics["r2"] < 0.75:
        result["warning"] = "low_model_confidence"

    return result
