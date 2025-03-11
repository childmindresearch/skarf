"""Adapt a covariance matrix as a linear VAR model via a polynomial fit."""

from copy import deepcopy
from numbers import Integral, Real
from typing import Literal, Self

import numpy as np
from numpy.random import RandomState
from scipy.linalg import block_diag
from sklearn.base import MetaEstimatorMixin, clone, _fit_context
from sklearn.utils.validation import validate_data
from sklearn.utils._param_validation import HasMethods, Interval, StrOptions
from sklearn.covariance import EmpiricalCovariance
from sklearn.utils.validation import check_is_fitted

from ._base import BaseVAR, _preprocess_data


class CovarianceVAR(MetaEstimatorMixin, BaseVAR):
    """Covariance based VAR model.

    Thsi model fits a linear VAR model parameterized by an underlying covariance matrix.
    The coefficients of the VAR model are represented as a learned polynomial of the
    covariance coefficients::

        A[l] = sum(b[l, i] * C ** (i + 1) for i in range(degree))

    for `l = 1, ..., order`, where `C` is the fixed covariance matrix and `b[l, i]` are
    the learned polynomial coefficients.

    Parameters
    ----------
    estimator : estimator object
        `Covariance` estimator object implementing `fit()` and having a `covariance_`
        attribute.

    order : int, default=1
        VAR model order, i.e. the number of past "lags" to include when predicting a
        future time point.

    lag : int, default=1
        Base temporal prediction lag/offset.

    degree : int, default=3
        Degree of the polynomial re-parameterization.

    use_precision : bool, default=False
        Use the covariance estimator's precision (inverse covariance) matrix.

    alpha : float, default=None
        Ridge regression penalty parameter on the polynomial coefficients.

    mode : {'full', 'leave_one_out'}, default='full'
        Model fit mode:

        * 'full' : Use the full covariance.

        * 'leave_one_out' : Zero the diagonal of the covariance before model fitting. This\
            prevents autocorrelation from improving model fit.

    random_state : int, RandomState instance, default=None
        The seed of the pseudo random number generator used when sampling.
        Note that using an int will produce identical results on each call to `sample`.
        Passing a `RandomState` instance will produce varying but reproducible sampling
        results.

    Attributes
    ----------
    coef_ : array of shape (order, n_targets, n_features)
        Estimated coefficients for the VAR model. The terms are ordered by increasing
        lag.  The `i`th row of each term contains the prediction coefficients for the
        `i`th feature.

    n_features_in_ : int
        Number of features seen during :term:`fit`.

    feature_names_in_ : array of shape (`n_features_in_`,)
        Names of features seen during `fit`. Defined only when `X` has feature names
        that are all strings.

    estimator_ : Estimator object
        Fit covariance estimator. If `frozen = True`, then the `estimator` parameter
        must already be fit, and a deep copy is made.

    beta_ : array of shape (order, degree)
        Array of polynomial coefficients.

    rank_ : int
        Rank of the polynomial regression design matrix.

    singular_ : array of shape (order * degree,)
        Singular values of the design matrix.
    """

    _parameter_constraints = {
        **BaseVAR._parameter_constraints,
        "estimator": [HasMethods(["fit"])],
        "degree": [Interval(Integral, 1, None, closed="left")],
        "use_precision": ["boolean"],
        "alpha": [Interval(Real, 0, None, closed="neither"), None],
        "mode": [StrOptions({"full", "leave_one_out"})],
    }

    estimator_: EmpiricalCovariance
    """Fit covariance estimator."""
    beta_: np.ndarray
    """Array of polynomial coefficients, shape (order, degree)."""
    rank_: int
    """Rank of the polynomial regression design matrix."""
    singular_: np.ndarray
    """Singular values of the design matrix, shape (order * degree,)"""

    def __init__(
        self,
        estimator: EmpiricalCovariance,
        order: int = 1,
        lag: int = 1,
        degree: int = 3,
        alpha: float | None = None,
        mode: Literal["full", "leave_one_out"] = "leave_one_out",
        use_precision: bool = False,
        frozen: bool = False,
        random_state: int | RandomState | None = None,
    ):
        super().__init__(order=order, lag=lag, random_state=random_state)
        self.estimator = estimator
        self.degree = degree
        self.alpha = alpha
        self.mode = mode
        self.use_precision = use_precision
        self.frozen = frozen

    @_fit_context(prefer_skip_nested_validation=False)
    def fit(
        self,
        X: np.ndarray,
        y: None = None,
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
        X = validate_data(self, X)
        X_stride, y_shift, _, sample_weight_shift, _ = _preprocess_data(
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
            X_stride = sample_weight_shift[:, None, None] * X_stride
            y_shift = sample_weight_shift[:, None] * y_shift

        if self.frozen:
            check_is_fitted(self.estimator)
            estimator = deepcopy(self.estimator)
            if estimator.covariance_.shape[1] != X.shape[1]:
                raise ValueError(
                    "Shape of frozen covariance estimator doesn't match input data X"
                )
        else:
            estimator = clone(self.estimator)
            estimator.fit(X)

        if self.use_precision:
            mat = estimator.get_precision()
        else:
            mat = estimator.covariance_
        mat = _preprocess_covariance(mat, with_diagonal=self.mode == "full")

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

        self.estimator_ = estimator
        self.beta_ = beta
        self.rank_ = rank
        self.singular_ = singular_values
        self.coef_ = coef
        return self

    def score(
        self,
        X: np.ndarray,
        y: None = None,
        segments: np.ndarray | None = None,
        sample_weight: np.ndarray | None = None,
    ) -> np.ndarray:
        """Return the prediction score for the model (by default R2).

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training multivariate time series.

        y : Ignored
            Ignored.

        segments : array-like of shape (n_samples,)
            Indicator array of contiguous temporal segments in `X`.

        sample_weight : float or array-like of shape (n_samples,), default=None
            Sample weights. Only binary sample weights indicating time points to
            include/exclude are currently supported.

        Returns
        -------
        score: float
            Mean VAR prediction score (by default R2, see `scoring_function`).
        """
        return super().score(X, y=None, segments=segments, sample_weight=sample_weight)


def _preprocess_covariance(
    covariance: np.ndarray, with_diagonal: bool = True
) -> np.ndarray:
    """Preprocess covariance matrix for VAR model."""
    assert (
        isinstance(covariance, np.ndarray)
        and covariance.ndim == 2
        and covariance.shape[0] == covariance.shape[1]
    ), "covariance matrix not valid"

    mat = np.where(np.isnan(covariance), 0.0, covariance)
    if not with_diagonal:
        np.fill_diagonal(mat, 0.0)
    mat = mat / (np.max(np.abs(mat)) + np.finfo(mat.dtype).eps)
    mat = np.ascontiguousarray(mat)
    return mat
