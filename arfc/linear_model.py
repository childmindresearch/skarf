from typing import TypeVar

import numpy as np
from sklearn.base import MetaEstimatorMixin, clone
from sklearn.linear_model import LinearRegression

from .base import ARModel
from . import timeseries as ts

T = TypeVar("T", bound="LinearARModel")


class LinearARModel(ARModel, MetaEstimatorMixin):
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

    def fit(self: T, X: ts.TimeseriesLike) -> T:
        X = ts.as_numpy(X)

        # X_stride: (order, time, dim)
        # X_shift: (time, dim)
        X_stride = self.tstride(X)
        X_shift = self.tshift(X)
        if ts.is_batch_timeseries(X):
            X_stride = np.concatenate(X_stride, axis=1)
            X_shift = np.concatenate(X_shift)
        dim = X_shift.shape[-1]

        if self.per_target:
            estimators = [
                self._fit_component(X_stride, X_shift, ii) for ii in range(dim)
            ]
            coef = np.stack([estimator.coef_ for estimator in estimators])
        else:
            estimator = self._fit_joint(X_stride, X_shift)
            coef = estimator.coef_

        # coef: (dim, order * dim)
        armats = np.ascontiguousarray(coef.reshape(dim, self.order, dim).swapaxes(0, 1))

        if not self.with_diagonal:
            armats[:, np.arange(dim), np.arange(dim)] = 0.0

        if self.per_target:
            self.estimators_ = estimators
        self.armats_ = armats
        return self

    def _fit_component(
        self: T, X_stride: np.ndarray, X_shift: np.ndarray, index: int
    ) -> LinearRegression:
        estimator = clone(self.estimator)
        if not self.with_diagonal:
            X_stride = X_stride.copy()
            X_stride[:, :, index] = 0
        X_stride_flat = self._flatten_strided(X_stride)
        estimator.fit(X_stride_flat, X_shift[:, index])
        return estimator

    def _fit_joint(
        self: T, X_stride: np.ndarray, X_shift: np.ndarray
    ) -> LinearRegression:
        X_stride_flat = self._flatten_strided(X_stride)
        self.estimator.fit(X_stride_flat, X_shift)
        return self.estimator

    def _predict_single(self: T, X: np.ndarray) -> np.ndarray:
        # predict using underlying models
        # should be equivalent to base prediction, but just to be careful
        # (one possible difference is intercept/scaling).
        X_stride = self.tstride(X)
        dim = X_stride.shape[-1]

        if self.per_target:
            X_pred = np.stack(
                [self._predict_component(X_stride, ii) for ii in range(dim)],
                axis=-1,
            )
        else:
            X_pred = self._predict_joint(X_stride)
        return X_pred

    def _predict_component(self: T, X_stride: np.ndarray, index: int) -> np.ndarray:
        if not self.with_diagonal:
            X_stride = X_stride.copy()
            X_stride[:, :, index] = 0
        X_stride_flat = self._flatten_strided(X_stride)
        X_pred_i = self.estimators_[index].predict(X_stride_flat)
        return X_pred_i

    def _predict_joint(self: T, X_stride: np.ndarray) -> np.ndarray:
        X_stride_flat = self._flatten_strided(X_stride)
        X_pred = self.estimator.predict(X_stride_flat)
        return X_pred

    def _flatten_strided(self: T, X_stride: np.ndarray) -> np.ndarray:
        assert X_stride.shape[0] == self.order, "invalid strided input shape"
        _, T, D = X_stride.shape
        return X_stride.swapaxes(0, 1).reshape((T, self.order * D))
