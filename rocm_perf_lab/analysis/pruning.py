def prune_configs(predictions, best_runtime, factor=1.75):
    return [
        i for i, pred in enumerate(predictions)
        if pred <= factor * best_runtime
    ]
