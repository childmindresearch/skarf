from typing import TypeVar

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.metrics import r2_score
from sklearn.utils.validation import check_is_fitted

from .utils import tsplit, group_tsplit


T = TypeVar("T", bound="ARModel")


class ARModel(BaseEstimator):
    armats_: np.ndarray

    def __init__(self: T, order: int = 1, lag: int = 1):
        self.order = order
        self.lag = lag

    def fit(self: T, X: np.ndarray, groups: np.ndarray | None = None) -> T:
        ...

    def predict(self: T, X_pres: np.ndarray) -> np.ndarray:
        check_is_fitted(self)
        assert X_pres.ndim == 3 and X_pres.shape[1] == self.order, "invalid X_pres"

        X_pred = sum(X_pres[:, step] @ self.armats_[step] for step in range(self.order))
        return X_pred

    def score(self: T, X: np.ndarray, groups: np.ndarray | None = None) -> float:
        X_pres, X_post, _ = self.tsplit(X, groups=groups)
        X_pred = self.predict(X_pres)
        score = r2_score(X_post, X_pred)
        return score

    def tsplit(
        self: T, X: np.ndarray, groups: np.ndarray | None = None
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if groups is None:
            X_pre, X_post = tsplit(X, order=self.order, lag=self.lag)
            split_groups = None
        else:
            X_pre, X_post, split_groups = group_tsplit(
                X, groups, order=self.order, lag=self.lag
            )
        return X_pre, X_post, split_groups
