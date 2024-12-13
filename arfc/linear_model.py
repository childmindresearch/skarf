import numpy as np
from sklearn.base import MetaEstimatorMixin, clone
from sklearn.linear_model import LinearRegression

from .base import ARModel
from . import timeseries as ts


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

    def fit(self, X: ts.TimeseriesLike) -> "LinearARModel":
        if ts.is_grouped_timeseries(X):
            groups = X.iloc[:, 0].values
        elif ts.is_batch_timeseries(X):
            groups = np.arange(len(X))
        else:
            groups = None

        if groups is not None and len(np.unique(groups)) == 1:
            groups = None

        # X_stride: (order, time, dim)
        # X_shift: (time, dim)
        X = ts.as_numpy(X)
        X_stride = self.tstride(X)
        X_shift = self.tshift(X)

        if groups is not None:
            groups = [
                np.full(len(X_shifti), group)
                for group, X_shifti in zip(groups, X_shift)
            ]
            groups = np.concatenate(groups)

        if ts.is_batch_timeseries(X):
            X_stride = np.concatenate(X_stride, axis=1)
            X_shift = np.concatenate(X_shift)
        dim = X_shift.shape[-1]

        if self.per_target:
            estimators = [
                self._fit_component(X_stride, X_shift, ii, groups=groups)
                for ii in range(dim)
            ]
            coef = np.stack([estimator.coef_ for estimator in estimators])
        else:
            estimator = self._fit_joint(X_stride, X_shift, groups=groups)
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
        self,
        X_stride: np.ndarray,
        X_shift: np.ndarray,
        index: int,
        groups: np.ndarray | None = None,
    ) -> LinearRegression:
        estimator = clone(self.estimator)
        if not self.with_diagonal:
            X_stride = X_stride.copy()
            X_stride[:, :, index] = 0
        X_stride_flat = self._flatten_strided(X_stride)
        return _try_fit_groups(
            estimator, X_stride_flat, X_shift[:, index], groups=groups
        )

    def _fit_joint(
        self,
        X_stride: np.ndarray,
        X_shift: np.ndarray,
        groups: np.ndarray | None = None,
    ) -> LinearRegression:
        X_stride_flat = self._flatten_strided(X_stride)
        return _try_fit_groups(self.estimator, X_stride_flat, X_shift, groups=groups)

    def _predict_single(self, X: np.ndarray) -> np.ndarray:
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

    def _predict_component(self, X_stride: np.ndarray, index: int) -> np.ndarray:
        if not self.with_diagonal:
            X_stride = X_stride.copy()
            X_stride[:, :, index] = 0
        X_stride_flat = self._flatten_strided(X_stride)
        X_pred_i = self.estimators_[index].predict(X_stride_flat)
        return X_pred_i

    def _predict_joint(self, X_stride: np.ndarray) -> np.ndarray:
        X_stride_flat = self._flatten_strided(X_stride)
        X_pred = self.estimator.predict(X_stride_flat)
        return X_pred

    def _flatten_strided(self, X_stride: np.ndarray) -> np.ndarray:
        assert X_stride.shape[0] == self.order, "invalid strided input shape"
        _, T, D = X_stride.shape
        return X_stride.swapaxes(0, 1).reshape((T, self.order * D))


def _try_fit_groups(
    model, X: np.ndarray, y: np.ndarray, groups: np.ndarray | None = None
):
    """
    Some estimators accept groups, others don't. In principle it's possible to determine
    what params an estimator accepts. But it's too much of a hassle.

    https://scikit-learn.org/1.5/auto_examples/miscellaneous/plot_metadata_routing.html
    """
    try:
        return model.fit(X, y, groups=groups)
    except TypeError:
        return model.fit(X, y)
