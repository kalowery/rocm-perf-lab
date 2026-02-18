import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression


class PerformanceRegressor:
    def __init__(self, degree: int = 2):
        self.degree = degree
        self.poly = PolynomialFeatures(degree=degree, include_bias=False)
        self.model = LinearRegression()
        self._fitted = False
        self._r2 = None
        self._residual_std = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        X_poly = self.poly.fit_transform(X)
        self.model.fit(X_poly, y)
        predictions = self.model.predict(X_poly)

        residuals = y - predictions
        ss_res = np.sum(residuals ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)

        self._r2 = 1 - ss_res / ss_tot if ss_tot != 0 else 0.0
        self._residual_std = float(np.std(residuals))
        self._fitted = True

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("Model not fitted.")
        X_poly = self.poly.transform(X)
        return self.model.predict(X_poly)

    def metrics(self):
        if not self._fitted:
            raise RuntimeError("Model not fitted.")
        return {
            "r2": float(self._r2),
            "residual_std": float(self._residual_std)
        }
