from typing import Self

import numpy as np
from scipy.linalg import block_diag
from sklearn.base import MetaEstimatorMixin
from sklearn.covariance import EmpiricalCovariance
from sklearn.utils.validation import check_is_fitted

from .base import BaseVAR, _preprocess_data


class CovarianceVAR(BaseVAR, MetaEstimatorMixin):
    beta_: np.ndarray
    rank_: int
    singular_: np.ndarray

    def __init__(
        self,
        estimator: EmpiricalCovariance,
        order: int = 1,
        lag: int = 1,
        degree: int = 3,
        alpha: float | None = None,
        use_precision: bool = False,
        frozen: bool = False,
    ):
        super().__init__(order=order, lag=lag)
        self.estimator = estimator
        self.degree = degree
        self.alpha = alpha
        self.use_precision = use_precision
        self.frozen = frozen

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray | None = None,
        segments: np.ndarray | None = None,
        sample_weight: np.ndarray | None = None,
    ) -> Self:
        """Fit the model with X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training multivariate time series.

        y : Ignored
            Not used, present here for API consistency by convention.

        segments : array-like of shape (n_samples,)
            Indicator array of contiguous temporal segments in `X`.

        sample_weight : array-like of shape (n_samples,), default=None
            Sample weights. Only binary sample weights indicating time points to
            include/exclude are currently supported.

        Returns
        -------
        self : object
            Returns the instance itself.
        """
        X_stride, y_shift, sample_weight_shift, _ = _preprocess_data(
            X,
            y=None,
            order=self.order,
            lag=self.lag,
            segments=segments,
            sample_weight=sample_weight,
        )

        # Mask time points that are excluded by sample weight.
        # Nb that only binary sample weight is supported.
        if sample_weight_shift is not None:
            X_stride = sample_weight_shift[None, :, None] * X_stride
            y_shift = sample_weight_shift[:, None] * y_shift

        if self.frozen:
            check_is_fitted(self.estimator)
            assert (
                self.estimator.covariance_.shape[1] == X.shape[1]
            ), "Shape of frozen covariance estimator doesn't match input data X"
        else:
            self.estimator.fit(X)

        if self.use_precision:
            mat = self.estimator.get_precision()
        else:
            mat = self.estimator.covariance_
        mat = _preprocess_covariance(mat)

        # pre-compute polynomial ar terms
        pow_mats = np.stack([mat**deg for deg in range(1, self.degree + 1)])
        A = np.stack(
            [
                (X_stride[:, step] @ pmat.T).flatten()
                for step in range(self.order)
                for pmat in pow_mats
            ],
            axis=1,
        )
        b = y_shift.flatten()

        # Augment for ridge penalty of reconstructed ar matrix. We want to penalize the
        # squared norm of each lag ar matrix, so we construct a block diagonal matrix of
        # the component matrices.
        if self.alpha:
            block = pow_mats.reshape((self.degree, -1)).T
            ridge_blocks = block_diag(*[block for step in range(self.order)])
            A = np.concatenate([A, np.sqrt(self.alpha) * ridge_blocks])
            b = np.concatenate([b, np.zeros(len(ridge_blocks))])

        beta, residuals, rank, singular_values = np.linalg.lstsq(A, b, rcond=-1)
        beta = beta.reshape((self.order, self.degree))
        coef = np.einsum("pq,qcd->pcd", beta, pow_mats)

        self.beta_ = beta
        self.rank_ = rank
        self.singular_ = singular_values
        self.coef_ = coef
        return self


def _preprocess_covariance(covariance: np.ndarray) -> np.ndarray:
    assert (
        isinstance(covariance, np.ndarray)
        and covariance.ndim == 2
        and covariance.shape[0] == covariance.shape[1]
    ), "covariance matrix not valid"

    mat = np.where(np.isnan(covariance), 0.0, covariance)
    np.fill_diagonal(mat, 0.0)  # ignore diagonal
    mat = mat / np.max(np.abs(mat))  # scale values to [-1, 1]
    mat = np.ascontiguousarray(mat)
    return mat
