import numpy as np
from sklearn.base import BaseEstimator
from sklearn.metrics import r2_score
from sklearn.utils.validation import check_is_fitted

from . import timeseries as ts


class ARModel(BaseEstimator):
    armats_: np.ndarray

    def __init__(self, order: int = 1, lag: int = 1):
        self.order = order
        self.lag = lag

    def fit(self, X: ts.TimeseriesLike) -> "ARModel":
        ...

    def predict(self, X: ts.TimeseriesLike) -> np.ndarray:
        check_is_fitted(self)
        X = ts.as_numpy(X)
        if ts.is_batch_timeseries(X):
            return ts.tstack([self._predict_single(Xi) for Xi in X])
        return self._predict_single(X)

    def _predict_single(self, X: np.ndarray) -> np.ndarray:
        assert ts.is_single_timeseries(
            X
        ), "input must be a single timeseries, shape (T, D)"
        X_stride = self.tstride(X)
        X_pred = sum(
            X_stride[step] @ self.armats_[step].T for step in range(self.order)
        )
        return X_pred

    def score(self, X: ts.TimeseriesLike) -> float:
        X = ts.as_numpy(X)
        X_pred = self.predict(X)
        X_shift = self.tshift(X)
        if ts.is_batch_timeseries(X):
            # Nb, for batch timeseries the metric is (macro-averaged) mean R2 over each
            # element timeseries, which is not necessarily identical to the global R2
            # over the concatenated timeseries.
            return np.mean(
                [self.scoring_function(X_shift[ii], X_pred[ii]) for ii in range(len(X))]
            )
        return self.scoring_function(X_shift, X_pred)

    def scoring_function(self, X_shift: np.ndarray, X_pred: np.ndarray) -> float:
        return r2_score(X_shift, X_pred)

    def tstride(self, X: ts.TimeseriesLike) -> np.ndarray:
        return ts.tstride(X, order=self.order, lag=self.lag)

    def tshift(self, X: ts.TimeseriesLike) -> np.ndarray:
        # account for the stride in higher-order AR prediction
        return ts.tshift(X, lag=self.order - 1 + self.lag)
