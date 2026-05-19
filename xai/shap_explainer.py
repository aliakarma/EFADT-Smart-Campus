"""
EFADT — SHAP Explainer (Proxy-Based)
======================================
Computes SHAP feature attributions for the occupancy forecast.

Approach (from paper):
  - Train a gradient-boosted proxy model to mimic the FL-LSTM predictions
  - Apply SHAP TreeExplainer (exact for tree-based models)
  - Return φⱼ values per feature for the current decision instance

Limitation acknowledged: SHAP is computed on the proxy, not the LSTM directly.
For direct LSTM SHAP, use shap.GradientExplainer with the trained OccupancyLSTM.
Reference: https://shap.readthedocs.io/en/latest/generated/shap.GradientExplainer.html

Decision latency budget: ~23ms for SHAP phase.
"""

from __future__ import annotations

import logging
import time
from typing import Optional

import numpy as np
import shap
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

FEATURE_NAMES = [
    "occupancy", "co2_ppm", "temperature_in", "temperature_out",
    "humidity", "hvac_power_kw", "hvac_setpoint", "motion_count",
    "hour_sin", "hour_cos", "day_of_week_sin", "day_of_week_cos",
    "month_sin", "month_cos",
]


class SHAPProxyExplainer:
    """
    Gradient-boosted proxy model for SHAP explanations of FL-LSTM predictions.

    Workflow:
    1. Collect (X, lstm_predictions) pairs during inference
    2. Fit GBM proxy to approximate LSTM mapping
    3. Use SHAP TreeExplainer to compute attributions per feature

    Parameters
    ----------
    n_estimators : int
        Number of GBM trees.
    max_depth : int
        Tree depth.
    learning_rate : float
    feature_names : list[str]
        Names of the 14 input features.
    """

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = 5,
        learning_rate: float = 0.1,
        feature_names: Optional[list[str]] = None,
    ) -> None:
        self.feature_names = feature_names or FEATURE_NAMES
        self.proxy = GradientBoostingRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            random_state=42,
        )
        self.explainer: Optional[shap.TreeExplainer] = None
        self._is_fitted = False
        self._X_background: Optional[np.ndarray] = None

    def fit(
        self,
        X_train: np.ndarray,
        lstm_predictions: np.ndarray,
    ) -> None:
        """
        Fit the gradient-boosted proxy model.

        Parameters
        ----------
        X_train : np.ndarray, shape (n_samples, lookback * n_features) or (n_samples, n_features)
            Flattened input features.
        lstm_predictions : np.ndarray, shape (n_samples,)
            LSTM model predictions (not ground truth — proxy mimics the LSTM).
        """
        if X_train.ndim > 2:
            X_flat = X_train.reshape(len(X_train), -1)
        else:
            X_flat = X_train

        logger.info(f"Fitting proxy GBM on {len(X_flat)} samples, {X_flat.shape[1]} features")
        t0 = time.time()

        self.proxy.fit(X_flat, lstm_predictions)
        self.explainer = shap.TreeExplainer(self.proxy)

        # Store background for SHAP baseline computation
        # Use 100 representative samples (or all if fewer)
        n_bg = min(100, len(X_flat))
        rng_bg = np.random.default_rng(42)
        idx = rng_bg.choice(len(X_flat), size=n_bg, replace=False)
        self._X_background = X_flat[idx]
        self._n_raw_features = X_flat.shape[1]
        self._is_fitted = True

        elapsed = time.time() - t0
        logger.info(f"Proxy fitted in {elapsed:.2f}s")

    def explain(
        self,
        x_instance: np.ndarray,
    ) -> np.ndarray:
        """
        Compute SHAP values for a single inference instance.

        Parameters
        ----------
        x_instance : np.ndarray
            Either (lookback, n_features) or (n_features,) for the current timestep.

        Returns
        -------
        np.ndarray, shape (n_features,) or (lookback * n_features,)
            SHAP φⱼ values for each feature.
        """
        if not self._is_fitted:
            raise RuntimeError("Call fit() before explain().")

        t0 = time.time()

        if x_instance.ndim > 1:
            x_flat = x_instance.flatten().reshape(1, -1)
        else:
            x_flat = x_instance.reshape(1, -1)

        shap_values = self.explainer.shap_values(x_flat)
        elapsed_ms = (time.time() - t0) * 1000

        logger.debug(f"SHAP computed in {elapsed_ms:.1f}ms")

        # If lookback-flattened, aggregate to per-feature level by summing over timesteps
        flat_shap = shap_values[0]   # shape: (lookback * n_features,)

        if len(flat_shap) > len(self.feature_names):
            # Reshape to (lookback, n_features) and sum over timesteps
            n_features = len(self.feature_names)
            lookback = len(flat_shap) // n_features
            reshaped = flat_shap[:lookback * n_features].reshape(lookback, n_features)
            per_feature = reshaped.sum(axis=0)
        else:
            per_feature = flat_shap[:len(self.feature_names)]

        return per_feature

    def get_top_features(
        self,
        shap_values: np.ndarray,
        k: int = 3,
    ) -> list[tuple[str, float]]:
        """
        Return top-k features by absolute SHAP magnitude.

        Parameters
        ----------
        shap_values : np.ndarray, shape (n_features,)
        k : int

        Returns
        -------
        list of (feature_name, shap_value) sorted by |φⱼ| descending.
        """
        indexed = list(zip(self.feature_names, shap_values))
        indexed.sort(key=lambda x: abs(x[1]), reverse=True)
        return indexed[:k]

    def predict_proxy(self, X: np.ndarray) -> np.ndarray:
        """Use the fitted proxy to make predictions (for debugging)."""
        if not self._is_fitted:
            raise RuntimeError("Call fit() first.")
        if X.ndim > 2:
            X = X.reshape(len(X), -1)
        return self.proxy.predict(X)

    def proxy_fidelity(
        self,
        X_val: np.ndarray,
        lstm_preds_val: np.ndarray,
    ) -> float:
        """
        Compute Spearman rank correlation between proxy and LSTM predictions.
        Higher = better proxy fidelity (SHF metric from Section 6).
        """
        from scipy.stats import spearmanr
        if not self._is_fitted:
            raise RuntimeError("Call fit() first.")
        proxy_preds = self.predict_proxy(X_val)
        corr, _ = spearmanr(proxy_preds, lstm_preds_val)
        return float(corr)

    def save(self, path: str) -> None:
        """Persist fitted proxy and explainer to disk."""
        import pickle
        if not self._is_fitted:
            raise RuntimeError("Explainer not fitted. Call fit() first.")
        with open(path, "wb") as f:
            pickle.dump({
                "proxy": self.proxy,
                "feature_names": self.feature_names,
                "n_raw_features": self._n_raw_features,
                "X_background": self._X_background,
            }, f)
        logger.info(f"SHAPProxyExplainer saved to {path}")

    @classmethod
    def load(cls, path: str) -> "SHAPProxyExplainer":
        """Load a previously fitted proxy explainer."""
        import pickle
        with open(path, "rb") as f:
            data = pickle.load(f)
        instance = cls(feature_names=data["feature_names"])
        instance.proxy = data["proxy"]
        instance.explainer = shap.TreeExplainer(instance.proxy)
        instance._n_raw_features = data["n_raw_features"]
        instance._X_background = data["X_background"]
        instance._is_fitted = True
        return instance


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Quick test
    rng = np.random.default_rng(42)
    n_samples = 500
    lookback, n_features = 12, 14
    X = rng.normal(0, 1, (n_samples, lookback, n_features))
    # Fake LSTM predictions (linear combination of features + noise)
    lstm_preds = (X[:, -1, :] @ rng.normal(0, 1, n_features)).flatten() + rng.normal(0, 2, n_samples)

    explainer = SHAPProxyExplainer()
    explainer.fit(X, lstm_preds)

    # Explain one instance
    x_test = rng.normal(0, 1, (lookback, n_features))
    shap_vals = explainer.explain(x_test)
    top3 = explainer.get_top_features(shap_vals, k=3)

    print(f"SHAP values shape: {shap_vals.shape}")
    print(f"Top-3 features: {top3}")
    print(f"SHF (proxy fidelity): {explainer.proxy_fidelity(X, lstm_preds):.3f}")
