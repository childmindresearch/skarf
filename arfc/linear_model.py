from typing import TypeVar

import numpy as np
from sklearn.base import clone
from sklearn.linear_model import LinearRegression
from sklearn.utils.validation import check_is_fitted

from .base import ARModel

T = TypeVar("T", bound="LinearARModel")


class LinearARModel(ARModel):
    def __init__(
        self,
        estimator: LinearRegression,
        order: int = 1,
        lag: int = 1,
        with_diagonal: bool = True,
        per_target: bool = False,
    ):
        super().__init__(order=order, lag=lag)
        assert (
            with_diagonal or per_target
        ), "excluding diagonal requires a per target linear model"
        self.estimator = estimator
        self.with_diagonal = with_diagonal
        self.per_target = per_target

    def fit(self: T, X: np.ndarray, groups: np.ndarray | None = None) -> T:
        # X_pres: (time, order, dim)
        # X_post: (time, dim)
        dim = X.shape[1]
        X_pres, X_post, _ = self.tsplit(X, groups=groups)

        if self.per_target:
            estimators = [self._fit_single(X_pres, X_post, ii) for ii in range(dim)]
            coef = np.stack([estimator.coef_ for estimator in estimators])
        else:
            estimator = self._fit_batch(X_pres, X_post)
            coef = estimator.coef_

        # coef: (dim, order * dim)
        armats = np.ascontiguousarray(
            coef.reshape(dim, self.order, dim).transpose(1, 0, 2)
        )

        if not self.with_diagonal:
            armats[:, np.arange(dim), np.arange(dim)] = 0.0

        if self.per_target:
            self.estimators_ = estimators
        self.armats_ = armats
        return self

    def _fit_single(
        self: T, X_pres: np.ndarray, X_post: np.ndarray, index: int
    ) -> LinearRegression:
        estimator = clone(self.estimator)
        if not self.with_diagonal:
            X_pres = X_pres.copy()
            X_pres[:, :, index] = 0
        X_pres_flat = X_pres.reshape((X_pres.shape[0], -1))
        estimator.fit(X_pres_flat, X_post[:, index])
        return estimator

    def _fit_batch(self: T, X_pres: np.ndarray, X_post: np.ndarray) -> LinearRegression:
        X_pres_flat = X_pres.reshape((X_pres.shape[0], -1))
        self.estimator.fit(X_pres_flat, X_post)
        return self.estimator

    def predict(self: T, X_pres: np.ndarray) -> np.ndarray:
        # predict using underlying models
        # should be equivalent to base prediction, but just to be careful
        # (one possible difference is intercept/scaling).
        check_is_fitted(self)
        assert X_pres.ndim == 3 and X_pres.shape[1] == self.order, "invalid X_pres"
        dim = X_pres.shape[2]

        if self.per_target:
            X_pred = np.stack(
                [self._predict_single(X_pres, ii) for ii in range(dim)],
                axis=-1,
            )
        else:
            X_pred = self._predict_batch(X_pres)
        return X_pred

    def _predict_single(self: T, X_pres: np.ndarray, index: int) -> np.ndarray:
        if not self.with_diagonal:
            X_pres = X_pres.copy()
            X_pres[:, :, index] = 0
        X_pres_flat = X_pres.reshape((X_pres.shape[0], -1))
        X_pred_i = self.estimators_[index].predict(X_pres_flat)
        return X_pred_i

    def _predict_batch(self: T, X_pres: np.ndarray) -> np.ndarray:
        X_pres_flat = X_pres.reshape((X_pres.shape[0], -1))
        X_pred = self.estimator.predict(X_pres_flat)
        return X_pred
