"""Adapt a covariance matrix as a linear VAR model via a polynomial fit."""

from copy import deepcopy
from numbers import Integral
from typing import Self

import numpy as np
from numpy.random import RandomState
from sklearn.base import MetaEstimatorMixin, clone, _fit_context
from sklearn.linear_model import ridge_regression
from sklearn.utils.validation import validate_data
from sklearn.utils._param_validation import HasMethods, Interval
from sklearn.covariance import EmpiricalCovariance

from ._base import BaseVAR, _preprocess_data


class CovarianceVAR(MetaEstimatorMixin, BaseVAR):
    """Covariance based VAR model.

    This model fits a linear VAR model parameterized by an underlying covariance matrix.
    The coefficients of the VAR model are represented as a learned polynomial of the
    covariance coefficients::

        A[l] = sum(b[l, i] * C ** i for i in range(degree + 1))

    for `l = 1, ..., order`, where `C` is the fixed covariance matrix and `b[l, i]` are
    the learned polynomial coefficients.

    Parameters
    ----------
    estimator : estimator object
        `Covariance` estimator object implementing `fit()` and having a `covariance_`
        attribute. If `frozen=True`, the estimator must already be fit.

    order : int, default=1
        VAR model order, i.e. the number of past "lags" to include when predicting a
        future time point.

    lag : int, default=1
        Base temporal prediction lag/offset.

    per_target : bool, default=False
        Fit separate polynomial model per target time series.

    degree : int, default=3
        Degree of the polynomial re-parameterization.

    use_precision : bool, default=False
        Use the covariance estimator's precision (inverse covariance) matrix.

    frozen : bool, default=False
        Reuse a previously fit covariance, rather than re-fit to the current data.

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
        Fit covariance estimator.

    beta_ : array of shape (order, degree + 1) or (n_features, order, degree + 1)
        Array of polynomial coefficients. One per feature if `per_target=True`.
    """

    _parameter_constraints = {
        **BaseVAR._parameter_constraints,
        "estimator": [HasMethods(["fit"])],
        "per_target": ["boolean"],
        "degree": [Interval(Integral, 1, None, closed="left")],
        "use_precision": ["boolean"],
        "frozen": ["boolean"],
    }

    estimator_: EmpiricalCovariance
    """Fit covariance estimator."""

    beta_: np.ndarray
    """Array of polynomial coefficients, shape (order, degree + 1)."""

    def __init__(
        self,
        estimator: EmpiricalCovariance,
        order: int = 1,
        lag: int = 1,
        degree: int = 3,
        per_target: bool = False,
        use_precision: bool = False,
        frozen: bool = False,
        random_state: int | RandomState | None = None,
    ):
        super().__init__(order=order, lag=lag, random_state=random_state)
        self.estimator = estimator
        self.degree = degree
        self.per_target = per_target
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
        X_stride, X_shift, _, sample_weight_shift, _ = _preprocess_data(
            X,
            y=None,
            order=self.order,
            lag=self.lag,
            segments=segments,
            sample_weight=sample_weight,
        )

        if self.frozen:
            if not hasattr(self.estimator, "covariance_"):
                raise ValueError(
                    "The input estimator must already be fit when frozen=True."
                )
            estimator = deepcopy(self.estimator)
            if estimator.covariance_.shape[1] != X.shape[1]:
                raise ValueError(
                    "Shape of pre-fit covariance estimator doesn't match input data X."
                )
        else:
            estimator = clone(self.estimator)
            estimator.fit(X)

        if self.use_precision:
            mat = estimator.get_precision()
        else:
            mat = estimator.covariance_
        mat = _preprocess_covariance(mat)

        coef, beta = self._fit_poly(
            mat, X_stride, X_shift, sample_weight=sample_weight_shift
        )

        self.estimator_ = estimator
        self.beta_ = beta
        self.coef_ = coef
        return self

    def _fit_poly(
        self,
        mat: np.ndarray,
        X_stride: np.ndarray,
        X_shift: np.ndarray,
        sample_weight: np.ndarray | None = None,
    ):
        n_samples, _, n_features = X_stride.shape

        # Polynomial VAR terms
        pow_mats = np.stack([mat**deg for deg in range(self.degree + 1)])
        # Set constant term diagonal to 0, since 0^0 = 1.
        np.fill_diagonal(pow_mats[0], 0.0)

        # (n_samples, n_features, (order * (degree + 1)))
        A = np.stack(
            [
                X_stride[:, step] @ pmat.T
                for step in range(self.order)
                for pmat in pow_mats
            ],
            axis=-1,
        )

        if self.per_target:
            betas = []
            for idx in range(n_features):
                beta = ridge_regression(
                    A[:, idx],
                    X_shift[:, idx],
                    alpha=0.0,
                    sample_weight=sample_weight,
                )
                betas.append(beta)
            # (n_features, order, degree + 1)
            beta = np.stack(betas)
            beta = beta.reshape(n_features, self.order, self.degree + 1)
            coef = np.einsum("mop,pmn->omn", beta, pow_mats)
        else:
            if sample_weight is not None:
                sample_weight = np.repeat(sample_weight, n_features)
            beta = ridge_regression(
                A.reshape(n_samples * n_features, -1),
                X_shift.flatten(),
                alpha=0.0,
                sample_weight=sample_weight,
            )
            beta = beta.reshape(self.order, self.degree + 1)
            coef = np.einsum("op,pmn->omn", beta, pow_mats)

        return coef, beta


def _preprocess_covariance(covariance: np.ndarray) -> np.ndarray:
    """Preprocess covariance matrix for VAR model."""
    assert (
        isinstance(covariance, np.ndarray)
        and covariance.ndim == 2
        and covariance.shape[0] == covariance.shape[1]
    ), "covariance matrix not valid"

    mat = np.where(np.isnan(covariance), 0.0, covariance)
    np.fill_diagonal(mat, 0.0)
    mat = mat / (np.max(np.abs(mat)) + np.finfo(mat.dtype).eps)
    mat = np.ascontiguousarray(mat)
    return mat
