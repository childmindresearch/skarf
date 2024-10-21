import warnings
from typing import TypeVar

import numpy as np
from scipy.linalg import block_diag
from sklearn.covariance import EmpiricalCovariance
from sklearn.utils.validation import check_is_fitted

from .base import ARModel


T = TypeVar("T", bound="CovarianceARModel")


class CovarianceARModel(ARModel):
    coef_: np.ndarray
    rank_: int
    singular_values_: np.ndarray

    def __init__(
        self,
        estimator: EmpiricalCovariance,
        order: int = 1,
        lag: int = 1,
        with_diagonal: bool = False,
        degree: int = 3,
        alpha: float | None = None,
        use_precision: bool = False,
        refit_cov: bool = True,
    ):
        super().__init__(order=order, lag=lag)
        self.estimator = estimator
        self.with_diagonal = with_diagonal
        self.degree = degree
        self.alpha = alpha
        self.use_precision = use_precision
        self.refit_cov = refit_cov

    def fit(self: T, X: np.ndarray, groups: np.ndarray | None = None) -> T:
        if self.refit_cov:
            self.estimator.fit(X)
        else:
            check_is_fitted(self.estimator)

        mat = self._get_precision() if self.use_precision else self._get_covariance()
        mat = self._preprocess_covariance(mat)

        X_pres, X_post, _ = self.tsplit(X, groups=groups)

        # pre-compute polynomial ar terms
        pow_mats = np.stack([mat**deg for deg in range(1, self.degree + 1)])
        A = np.stack(
            [
                (X_pres[:, step] @ pmat.T).flatten()
                for step in range(self.order)
                for pmat in pow_mats
            ],
            axis=-1,
        )
        b = X_post.flatten()

        # Augment for ridge penalty of reconstructed ar matrix. We want to penalize the
        # squared norm of each lag ar matrix, so we construct a block diagonal matrix of
        # the component matrices.
        if self.alpha:
            block = pow_mats.reshape((self.degree, -1)).T
            ridge_blocks = block_diag(*[block for step in range(self.order)])
            A = np.concatenate([A, np.sqrt(self.alpha) * ridge_blocks])
            b = np.concatenate([b, np.zeros(len(ridge_blocks))])

        coef, residuals, rank, singular_values = np.linalg.lstsq(A, b, rcond=-1)
        coef = coef.reshape((self.order, self.degree))

        armats = np.stack(
            [
                sum(
                    coef[step, deg - 1] * (mat**deg)
                    for deg in range(1, self.degree + 1)
                )
                for step in range(self.order)
            ]
        )

        self.coef_ = coef
        self.rank_ = rank
        self.singular_values_ = singular_values
        self.armats_ = armats
        return self

    def _get_covariance(self) -> np.ndarray:
        return self.estimator.covariance_

    def _get_precision(self) -> np.ndarray:
        return self.estimator.get_precision()

    def _preprocess_covariance(self, covariance: np.ndarray) -> np.ndarray:
        assert (
            isinstance(covariance, np.ndarray)
            and covariance.ndim == 2
            and covariance.shape[0] == covariance.shape[1]
        ), "covariance matrix not valid"

        mat = covariance.copy()
        mat[np.isnan(mat)] = 0.0  # ignore NaN
        if not self.with_diagonal:
            np.fill_diagonal(mat, 0.0)  # ignore diagonal
        mat = mat / max(np.max(np.abs(mat)), 1e-4)  # scale values
        mat = np.ascontiguousarray(mat)
        return mat


class FrozenCovariance(EmpiricalCovariance):
    def __init__(self, covariance: np.ndarray, store_precision: bool = True):
        super().__init__(store_precision=store_precision)
        self._set_covariance(covariance)

    def fit(self, X: np.ndarray, y: np.ndarray | None = None):
        warnings.warn("Calling fit has no effect on frozen covariance", RuntimeWarning)
        return self
