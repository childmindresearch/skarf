from itertools import repeat
from typing import Any, Self

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, MetaEstimatorMixin, TransformerMixin, clone
from sklearn.utils.validation import check_is_fitted
from sklearn.utils.parallel import Parallel, delayed

from .base import BaseVAR


class MultiVAR(BaseEstimator, MetaEstimatorMixin, TransformerMixin):
    estimators_: dict[int, BaseVAR]

    def __init__(self, estimator: BaseVAR, n_jobs: int | None = None):
        self.estimator = estimator
        self.n_jobs = n_jobs

    def fit(
        self,
        X: np.ndarray | pd.Series,
        y: np.ndarray | None = None,
        sample_ids: np.ndarray | None = None,
        **params,
    ) -> Self:
        X, sample_ids = _check_X_sample_ids(X, sample_ids)

        params_values = list(params.values())

        jobs = []
        for X_i, y_i, *params_values_i in _optional_zip(X, y, *params_values):
            params_i = {k: v for k, v in zip(params, params_values_i)}
            jobs.append(delayed(self._fit_single)(X_i, y=y_i, **params_i))

        results = Parallel(n_jobs=self.n_jobs)(jobs)
        self.estimators_ = {
            sample_id: est for sample_id, est in zip(sample_ids, results)
        }
        return self

    def _fit_single(
        self,
        X: np.ndarray,
        y: np.ndarray | None,
        **params,
    ) -> BaseVAR:
        estimator = clone(self.estimator)
        params = {k: v for k, v in params.items() if v is not None}
        estimator.fit(X, y=y, **params)
        return estimator

    def predict(
        self,
        X: np.ndarray | pd.Series,
        sample_ids: np.ndarray | None = None,
    ) -> np.ndarray:
        check_is_fitted(self)
        X, sample_ids = _check_X_sample_ids(X, sample_ids)

        X_pred = []
        for sample_id, X_i in zip(sample_ids, X):
            X_pred_i = self.estimators_[sample_id].predict(X_i)
            X_pred.append(X_pred_i)

        X_pred = stack_arrays(X_pred)
        return X_pred

    def score(
        self,
        X: np.ndarray | pd.Series,
        y: np.ndarray | None = None,
        sample_ids: np.ndarray | None = None,
        segments: np.ndarray | None = None,
        sample_weight: np.ndarray | None = None,
    ) -> float:
        check_is_fitted(self)
        X, sample_ids = _check_X_sample_ids(X, sample_ids)

        scores, lengths = [], []
        for sample_id, X_i, y_i, segments_i, sample_weight_i in _optional_zip(
            sample_ids, X, y, segments, sample_weight
        ):
            score = self.estimators_[sample_id].score(
                X=X_i,
                y=y_i,
                segments=segments_i,
                sample_weight=sample_weight_i,
            )
            scores.append(score)
            lengths.append(len(X_i))

        scores = np.array(scores)
        lengths = np.array(lengths)
        score = np.sum(scores * lengths) / np.sum(lengths)
        return score

    def transform(
        self, X: np.ndarray, sample_ids: np.ndarray | None = None
    ) -> np.ndarray:
        X, sample_ids = _check_X_sample_ids(X, sample_ids)
        coefs = np.stack(
            [
                self._transform_single(sample_id, X_i)
                for sample_id, X_i in zip(sample_ids, X)
            ]
        )
        return coefs

    def _transform_single(self, sample_id: int, X_i: np.ndarray) -> np.ndarray:
        # fit a new transformation for any unseen samples
        if sample_id not in self.estimators_:
            self.estimators_[sample_id] = self._fit_single(X_i)
        return self.estimators_[sample_id].coef_.copy()


def stack_arrays(arrays: list[np.ndarray]) -> np.ndarray:
    """Stack arrays of possibly different dimensions."""
    try:
        return np.stack(arrays)
    except ValueError:
        # https://stackoverflow.com/a/68824867
        stacked = np.empty(len(arrays), dtype=object)
        stacked[:] = arrays
        return stacked


def _optional_zip(*arrays):
    """Zip a sequence of iterables, repeating None for any that are None."""
    array = arrays[0]
    assert array is not None, "first array should not be None"
    length = len(array)

    arrays = [repeat(None, length) if arr is None else arr for arr in arrays]
    yield from zip(*arrays)


def _check_X_sample_ids(
    X: np.ndarray | pd.Series, sample_ids: np.ndarray | None
) -> tuple[np.ndarray, np.ndarray]:
    """Check input X and sample IDs.

    If X is a pandas Series and sample_ids is None the series index is used as the
    sample ID.
    """
    if _is_series_like(X):
        if sample_ids is None:
            sample_ids = np.asanyarray(X.index)
        X = np.asanyarray(X.values)
    elif sample_ids is None:
        sample_ids = np.arange(len(X))
    return X, sample_ids


def _is_series_like(obj: Any) -> bool:
    """Check if an object is a pandas Series or similar."""
    return hasattr(obj, "index") and hasattr(obj, "values")
