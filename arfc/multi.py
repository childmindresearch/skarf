import warnings
from typing import Any

import numpy as np
from sklearn.base import BaseEstimator, MetaEstimatorMixin, TransformerMixin, clone
from sklearn.utils.validation import check_is_fitted
from sklearn.utils.parallel import Parallel, delayed

from .base import ARModel
from . import timeseries as ts


class MultiARModel(BaseEstimator, MetaEstimatorMixin):
    estimators_: dict[int, ARModel]

    def __init__(self, estimator: ARModel, n_jobs: int | None = None):
        self.estimator = estimator
        self.n_jobs = n_jobs

    def fit(self, X: ts.TimeseriesLike, y: Any | None = None) -> "MultiARModel":
        jobs, groups = [], []
        for group, X_group in ts.iter_groups(X):
            groups.append(group)
            jobs.append(delayed(self._fit_single)(X_group))
        results = Parallel(n_jobs=self.n_jobs)(jobs)
        self.estimators_ = {group: est for group, est in zip(groups, results)}

        if len(self.estimators_) == 0:
            raise ValueError("No time series groups in training data.")

        if len(self.estimators_) == 1:
            warnings.warn(
                "Only one time series group. MultiARModel is "
                "intended for grouped time series.",
                RuntimeWarning,
            )
        return self

    def _fit_single(self, X: np.ndarray) -> ARModel:
        estimator = clone(self.estimator)
        estimator.fit(X)
        return estimator

    def predict(self, X: ts.TimeseriesLike) -> np.ndarray:
        check_is_fitted(self)

        groups, X_preds = [], []
        for group, X_group in ts.iter_groups(X):
            X_pred = self.estimators_[group].predict(X_group)
            groups.append(group)
            X_preds.append(X_pred)

        if ts.is_grouped_timeseries(X):
            # TODO: if there are extra columns in X, they will be dropped
            X_pred = ts.stack_groups(zip(groups, X_preds), column_name=X.columns[0])
        else:
            X_pred = ts.tstack(X_preds)
        return X_pred

    def score(self, X: ts.TimeseriesLike) -> float:
        # Nb, macro averaging
        check_is_fitted(self)
        score = np.mean(
            [
                self.estimators_[group].score(X_group)
                for group, X_group in ts.iter_groups(X)
            ]
        )
        return score


class MultiARTransformer(MultiARModel, TransformerMixin):
    def transform(self, X: ts.TimeseriesLike) -> np.ndarray:
        armats = np.stack(
            [
                self._transform_single(group, X_group)
                for group, X_group in ts.iter_groups(X)
            ]
        )
        return armats

    def _transform_single(self, group: int, X: ts.TimeseriesLike) -> np.ndarray:
        # fit a new transformation for any unseen samples
        if group not in self.estimators_:
            self.estimators_[group] = self._fit_single(X)
        return self.estimators_[group].armats_.copy()
