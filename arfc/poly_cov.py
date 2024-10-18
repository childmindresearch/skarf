from typing import TypeVar

import numpy as np
from sklearn.covariance import EmpiricalCovariance
from sklearn.utils.validation import check_is_fitted

from .base import ARModel


T = TypeVar("T", bound="PolyCovARModel")


class PolyCovARModel(ARModel):
    coef_: np.ndarray
    rank_: int
    singular_values_: np.ndarray

    def __init__(
        self,
        cov_estimator: EmpiricalCovariance,
        use_precision: bool = False,
        degree: int = 3,
        order: int = 1,
        lag: int = 1,
        refit_cov: bool = True,
    ):
        super().__init__(order=order, lag=lag)
        self.cov_estimator = cov_estimator
        self.use_precision = use_precision
        self.degree = degree
        self.refit_cov = refit_cov

    def fit(self: T, X: np.ndarray, groups: np.ndarray | None = None) -> T:
        if self.refit_cov:
            self.cov_estimator.fit(X)
        else:
            check_is_fitted(self.cov_estimator)

        mat = self.get_precision() if self.use_precision else self.get_covariance()
        mat = self._preprocess_covariance(mat)

        X_pres, X_post, _ = self.tsplit(X, groups=groups)

        # pre-compute polynomial terms
        # (ar_order * poly_order, tpts, dim)
        poly_terms = np.stack(
            [
                X_pres[:, step] @ (mat ** deg)
                for step in range(self.order)
                for deg in range(1, self.degree + 1)
            ]
        )
        A = poly_terms.reshape(poly_terms.shape[0], -1).T
        b = X_post.flatten()

        coef, residuals, rank, singular_values = np.linalg.lstsq(A, b, rcond=-1)
        coef = coef.reshape((self.order, self.degree))

        armats = np.stack(
            [
                sum(
                    coef[step, deg - 1] * (mat ** deg)
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

    def get_covariance(self) -> np.ndarray:
        return self.cov_estimator.covariance_

    def get_precision(self) -> np.ndarray:
        return self.cov_estimator.get_precision()

    def _preprocess_covariance(self, covariance: np.ndarray) -> np.ndarray:
        assert (
            isinstance(covariance, np.ndarray)
            and covariance.ndim == 2
            and covariance.shape[0] == covariance.shape[1]
        ), f"covariance matrix not valid"

        mat = covariance.copy()

        np.fill_diagonal(mat, 0.0)  # ignore diagonal
        mat[np.isnan(mat)] = 0.0    # ignore NaN
        mat = mat / max(np.max(np.abs(mat)), 1e-4)  # scale values
        mat = np.ascontiguousarray(mat.T)   # transpose (assuming i -> j direction)
        return mat
