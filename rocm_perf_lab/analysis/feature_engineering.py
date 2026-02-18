import numpy as np


class FeatureVectorizer:
    def __init__(self, feature_order: list[str]):
        self.feature_order = feature_order

    def transform(self, feature_dicts: list[dict]) -> np.ndarray:
        matrix = []
        for features in feature_dicts:
            row = []
            for key in self.feature_order:
                if key not in features:
                    raise ValueError(f"Missing feature: {key}")
                row.append(features[key])
            matrix.append(row)
        return np.array(matrix, dtype=float)
