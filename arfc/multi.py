from typing import TypeVar

import numpy as np
from sklearn.base import MetaEstimatorMixin, clone
from sklearn.utils.validation import check_is_fitted
from sklearn.metrics import r2_score

from .base import ARModel
from .utils import is_contiguous, iter_groups


T = TypeVar("T", bound="MultiARModel")


class MultiARModel(MetaEstimatorMixin):
    estimators_: dict[int, ARModel]

    def __init__(self: T, estimator: ARModel):
        self.estimator = estimator

    def fit(self: T, X: np.ndarray, groups: np.ndarray) -> T:
        estimators = {}
        for group, mask in iter_groups(groups):
            assert is_contiguous(mask), "groups are not temporally contiguous"
            estimators[group] = self.fit_single(X[mask])
        self.estimators_ = estimators
        return self

    def fit_single(self: T, X: np.ndarray) -> ARModel:
        estimator = clone(self.estimator)
        estimator.fit(X)
        return estimator

    def predict(self: T, X_pres: np.ndarray, groups: np.ndarray) -> np.ndarray:
        check_is_fitted(self)
        X_pred = np.zeros((X_pres.shape[0], X_pres.shape[2]))
        for group, mask in iter_groups(groups):
            X_pred[mask] = self.estimators_[group].predict(X_pres[mask])
        return X_pred

    def score(self: T, X: np.ndarray, groups: np.ndarray) -> float:
        X_pres, X_post, split_groups = self.estimator.tsplit(X, groups=groups)
        X_pred = self.predict(X_pres, groups=split_groups)
        score = r2_score(X_post, X_pred)
        return score


class MultiARTransformer:
    def __init__(self, multi_ar: MultiARModel):
        self.multi_ar = multi_ar

    def fit_transform(self, X: np.ndarray, sample_ids: np.ndarray | None) -> np.ndarray:
        assert sample_ids is None or sample_ids.shape == (len(X),), "invalid ids"
        X_flat, groups = flatten_sequences(X)
        if sample_ids is not None:
            groups = sample_ids[groups]
        self.multi_ar.fit(X_flat, groups)
        return self.transform(X)

    def transform(self, X: np.ndarray, sample_ids: np.ndarray | None) -> np.ndarray:
        if sample_ids is None:
            sample_ids = np.arange(len(X))
        armats = np.stack(
            [
                self.transform_single(X[ii], sample_id)
                for ii, sample_id in enumerate(sample_ids)
            ]
        )
        return armats

    def transform_single(self, X: np.ndarray, sample_id: int) -> np.ndarray:
        # fit a new transformation for any unseen samples
        if sample_id not in self.multi_ar.estimators_:
            self.multi_ar.estimators_[sample_id] = self.multi_ar.fit_single(X)
        return self.multi_ar.estimators_[sample_id].armats_.copy()


def flatten_sequences(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    assert np.atleast_3d(X), "expected X at least 3d"
    N, T = X.shape[:2]
    indices = np.repeat(np.arange(N), T)
    X = X.reshape((N * T, *X.shape[2:]))
    return X, indices
