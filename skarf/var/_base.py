"""Abstract base estimator class for VAR models."""

from abc import ABCMeta, abstractmethod
from numbers import Integral
from typing import NamedTuple, Self

import numpy as np
from numpy.random import RandomState
from sklearn.base import BaseEstimator
from sklearn.metrics import r2_score
from sklearn.utils.validation import (
    check_array,
    check_is_fitted,
    check_random_state,
    validate_data,
    _check_sample_weight,
)
from sklearn.utils._param_validation import Interval

from ._utils import _tstride, _segments_to_windows, _check_segments


class BaseVAR(BaseEstimator, metaclass=ABCMeta):
    """Base VAR model.

    Parameters
    ----------
    order : int, default=1
        VAR model order, i.e. the number of past "lags" to include when predicting a
        future time point.

    lag : int, default=1
        Base temporal prediction lag/offset.

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
    """

    _parameter_constraints = {
        "order": [Interval(Integral, 1, None, closed="left")],
        "lag": [Interval(Integral, 0, None, closed="left")],
        "random_state": ["random_state"],
    }

    coef_: np.ndarray
    """Array of VAR coefficients of shape (order, n_targets, n_features)."""

    n_features_in_: int
    """Number of features seen during `fit`."""

    feature_names_in_: np.ndarray
    """Names of features seen during `fit`. Defined only when `X` has feature names."""

    scoring_function = staticmethod(r2_score)
    """Static scoring function (default `r2_score`)"""

    def __init__(
        self,
        order: int = 1,
        lag: int = 1,
        random_state: int | RandomState | None = None,
    ):
        self.order = order
        self.lag = lag
        self.random_state = random_state

    @abstractmethod
    def fit(
        self,
        X: np.ndarray,
        y: None = None,
        segments: np.ndarray | None = None,
        sample_weight: np.ndarray | None = None,
        **params,
    ) -> Self:
        """Placeholder for fit. Subclasses should implement this method!

        Fit the model with X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training multivariate time series.

        y : Ignored
            Not used, present here for API consistency by convention.

        segments : array-like of shape (n_samples,)
            Indicator array of contiguous temporal segments in `X`.

        sample_weight : float or array-like of shape (n_samples,), default=None
            Sample weights. Only binary sample weights indicating time points to
            include/exclude are currently supported.

        Returns
        -------
        self : object
            Returns the instance itself.
        """

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict time series values for next time steps.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input multivariate time series.

        Returns
        -------
        X_pred : array-like of shape (n_samples, n_features)
            Next time step predictions, same shape as ``X``.
        """
        check_is_fitted(self)
        X = validate_data(self, X, reset=False)
        X_stride = _tstride(X, order=self.order, mode="same")
        return self._predict_strided(X_stride)

    def _predict_strided(self, X_stride: np.ndarray) -> np.ndarray:
        X_pred = np.einsum("npd,pkd->nk", X_stride, self.coef_)
        return X_pred

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
            Not used, present here for API consistency by convention.

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
        check_is_fitted(self)
        X = validate_data(self, X, reset=False)

        X_stride, X_shift, _, sample_weight_shift, _ = _preprocess_data(
            X,
            y=None,
            order=self.order,
            lag=self.lag,
            segments=segments,
            sample_weight=sample_weight,
        )
        X_pred = self._predict_strided(X_stride)
        score = self.scoring_function(
            X_shift, X_pred, sample_weight=sample_weight_shift
        )
        return score

    def sample(self, n_samples: int, X_init: np.ndarray | None = None) -> np.ndarray:
        """Sample simulated data from the VAR model.

        Parameters
        ----------
        n_samples : int
            Number of samples to generate

        X_init : array-like of shape (n_init_samples, n_features) or None
            Initial time series prefix. If `None`, then a random initial vector sampled
            from a standard Gaussian distribution is used.

        Returns
        -------
        X_samples : array-like of shape (n_samples, n_features)
        """
        check_is_fitted(self)
        if X_init is not None:
            X_init = validate_data(self, X_init, reset=False)

        if self.coef_.shape[1] != self.coef_.shape[2]:
            raise RuntimeError("Sampling requires n_targets == n_features.")
        if self.lag == 0:
            raise RuntimeError("Sampling not supported for lag 0.")

        if X_init is None:
            rng = check_random_state(self.random_state)
            X_init = rng.randn(self.order, self.coef_.shape[1])

        X_samples = X_init
        for _ in range(n_samples):
            X_next = self.predict(X_samples[-self.order :])[-1:]
            X_samples = np.concatenate([X_samples, X_next])

        X_samples = X_samples[-n_samples:]
        return X_samples


class _VARData(NamedTuple):
    X_stride: np.ndarray
    """Strided input data, shape (n_samples - order - lag + 1, order, n_features)."""
    y_shift: np.ndarray
    """Temporally shifted targets, shape (n_samples - order - lag + 1, n_targets)."""
    segments_shift: np.ndarray | None
    """Shifted temporal segments."""
    sample_weight_shift: np.ndarray | None
    """Shifted sample weights, shape (n_samples - order - lag + 1,)."""
    groups_shift: np.ndarray | None
    """Shifted CV data groups, shape (n_samples - order - lag + 1,)."""


def _preprocess_data(
    X: np.ndarray,
    y: np.ndarray | None = None,
    *,
    order: int = 1,
    lag: int = 1,
    segments: np.ndarray | None = None,
    sample_weight: np.ndarray | None = None,
    groups: np.ndarray | None = None,
) -> _VARData:
    """Preprocess data for VAR model fitting.

    Given input data X, shape (n_samples, n_features), and optional targets y, shape
    (n_samples, n_targets), Selects strided slices of input X_stride, shape (n_samples -
    order - lag + 1, order, n_features), and temporally shifted targets y_shift, shape
    (n_samples - order - lag + 1, n_targets).

    Also returns shifted sample weights and CV data groups, if provided. Only binary
    sample weights are supported.

    Sample weights are also temporally expanded so that all samples whose sliding
    prediction window overlaps with an excluded sample are also excluded.
    """
    min_samples = order + lag + 1
    X = check_array(X, ensure_min_samples=min_samples)

    if segments is not None:
        segments = _check_segments(segments, X=X)
    if sample_weight is not None:
        sample_weight = _check_sample_weight(sample_weight, X=X)
    if sample_weight is not None:
        if not np.allclose(sample_weight, sample_weight > 0):
            raise ValueError("Only binary sample_weight supported.")

    if y is None:
        y = X

    if segments is not None:
        windows, segment_values = _segments_to_windows(segments)
    else:
        windows, segment_values = [(0, len(X))], [0]

    X_stride, y_shift = [], []
    segments_shift = [] if segments is not None else None
    groups_shift = [] if groups is not None else None
    sample_weight_shift = [] if sample_weight is not None else None

    for (start, stop), value in zip(windows, segment_values):
        X_stride_i, y_shift_i = _align_X_y(
            X[start:stop], y[start:stop], order=order, lag=lag
        )

        X_stride.append(X_stride_i)
        y_shift.append(y_shift_i)

        if segments is not None:
            segments_shift.append(np.full(len(y_shift_i), value))

        # Mask out time points that have overlap with the excluded time points
        if sample_weight is not None:
            sample_weight_stride_i, sample_weight_shift_i = _align_X_y(
                sample_weight[start:stop], order=order, lag=lag
            )
            sample_weight_shift_i = np.concatenate(
                [sample_weight_stride_i, sample_weight_shift_i[:, None]], axis=1
            )
            sample_weight_shift_i = np.min(sample_weight_shift_i, axis=1)
            sample_weight_shift.append(sample_weight_shift_i)

        # Shift groups indicator to align with targets
        if groups is not None:
            groups_i = groups[start:stop]
            n_groups = len(np.unique(groups_i))
            if n_groups != 1:
                raise ValueError("Each data segment should contain only one CV group")
            groups_shift_i = groups_i[order + lag - 1 :]
            groups_shift.append(groups_shift_i)

    X_stride = np.concatenate(X_stride)
    y_shift = np.concatenate(y_shift)

    if segments is not None:
        segments_shift = np.concatenate(segments_shift)

    if sample_weight is not None:
        sample_weight_shift = np.concatenate(sample_weight_shift)

    if groups is not None:
        groups_shift = np.concatenate(groups_shift)

    return _VARData(
        X_stride, y_shift, segments_shift, sample_weight_shift, groups_shift
    )


def _align_X_y(
    X: np.ndarray, y: np.ndarray | None = None, order: int = 1, lag: int = 1
) -> tuple[np.ndarray, np.ndarray]:
    """Stride X and shift Y so that the two are aligned for VAR model fitting, with the
    given order and lag.

    - X: (n_samples, n_features)
    - y: (n_samples, n_targets)
    - X_stride: (n_samples - order - lag + 1, order, n_features)
    - y_shift: (n_samples - order - lag + 1, n_targets)
    """
    assert order > 0, "expected order > 0"
    assert lag >= 0, "expected lag >= 0"

    if y is None:
        y = X
    X_stride = _tstride(X, order=order, mode="valid")
    X_stride = X_stride[: len(X_stride) - lag]
    y_shift = y[order + lag - 1 :]
    return X_stride, y_shift
