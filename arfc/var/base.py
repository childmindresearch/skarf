from abc import ABCMeta, abstractmethod
from typing import Literal, NamedTuple, Self

import numpy as np
from numpy.random import RandomState
from sklearn.base import BaseEstimator
from sklearn.metrics import r2_score
from sklearn.utils.validation import check_is_fitted, check_random_state


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
        Pass an int for reproducible output across multiple function calls.

    Attributes
    ----------
    coef_ : array of shape (order, n_targets, n_features)
        Estimated coefficients for the VAR model. The terms are ordered by increasing
        lag.  The `i`th row of each term contains the prediction coefficients for the
        `i`th feature.
    """

    coef_: np.ndarray
    """Array of VAR coefficients of shape (order, n_targets, n_features)."""

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
        y: np.ndarray | None = None,
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

        y : array-like of shape (n_samples, n_targets,) or (n_samples,) or None
            Target time series. If `None`, the data itself is used as the target.

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
        """
        check_is_fitted(self)
        X_stride = _tstride(X, order=self.order, mode="same")
        return self._predict_strided(X_stride)

    def _predict_strided(self, X_stride: np.ndarray) -> np.ndarray:
        X_pred = np.einsum("npd,pkd->nk", X_stride, self.coef_)
        return X_pred

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
        assert (
            self.coef_.shape[1] == self.coef_.shape[2]
        ), "sampling requires n_targets == n_features"

        if X_init is None:
            rng = check_random_state(self.random_state)
            X_init = rng.randn((1, self.coef_.shape[1]))

        X_samples = X_init
        for _ in range(n_samples):
            X_next = self.predict(X_samples[-self.order :])[-1:]
            X_samples = np.concatenate([X_samples, X_next])

        X_samples = X_samples[-n_samples:]
        return X_samples

    def score(
        self,
        X: np.ndarray,
        y: np.ndarray | None = None,
        segments: np.ndarray | None = None,
        sample_weight: np.ndarray | None = None,
    ) -> np.ndarray:
        X_stride, y_shift, sample_weight_shift, _ = _preprocess_data(
            X,
            y=y,
            order=self.order,
            lag=self.lag,
            segments=segments,
            sample_weight=sample_weight,
        )
        X_pred = self._predict_strided(X_stride)

        return self.scoring_function(y_shift, X_pred, sample_weight=sample_weight_shift)

    scoring_function = r2_score


class _VARData(NamedTuple):
    X_stride: np.ndarray
    """Strided input data, shape (n_samples - order - lag + 1, order, n_features)."""
    y_shift: np.ndarray
    """Temporally shifted targets, shape (n_samples - order - lag + 1, n_targets)."""
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
    if sample_weight is not None:
        assert np.allclose(
            sample_weight, sample_weight > 0
        ), "only binary sample_weight supported"

    if y is None:
        y = X

    if segments is not None:
        _, windows = _segments_to_windows(segments)
    else:
        windows = [(0, len(X))]

    X_stride, y_shift = [], []
    groups_shift = [] if groups is not None else None
    sample_weight_shift = [] if sample_weight is not None else None

    for start, stop in windows:
        X_stride_i, y_shift_i = _align_X_y(
            X[start:stop], y[start:stop], order=order, lag=lag
        )

        X_stride.append(X_stride_i)
        y_shift.append(y_shift_i)

        # Shift groups indicator to align with targets
        if groups is not None:
            groups_i = groups[start:stop]
            n_groups = len(np.unique(groups_i))
            assert n_groups == 1, "expected each segment to contain exactly 1 group"
            groups_shift_i = groups_i[order - 1 + lag :]
            groups_shift.append(groups_shift_i)

        # Mask out time points that have overlap with the excluded time points
        if sample_weight is not None:
            sample_weight_stride_i, sample_weight_shift_i = _align_X_y(
                sample_weight[start:stop], order=order, lag=lag
            )
            sample_weight_shift_i = np.concatenate(
                [sample_weight_stride_i, sample_weight_shift_i[:, None]], axis=1
            )
            sample_weight_shift_i = np.minimum(sample_weight_shift_i, axis=1)
            sample_weight_shift.append(sample_weight_shift_i)

    X_stride = np.concatenate(X_stride)
    y_shift = np.concatenate(y_shift)

    if groups_shift is not None:
        groups_shift = np.concatenate(groups_shift)

    if sample_weight_shift is not None:
        sample_weight_shift = np.concatenate(sample_weight_shift)

    return _VARData(X_stride, y_shift, sample_weight_shift, groups_shift)


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
    if y is None:
        y = X
    # shape (n_samples - order + 1, order, n_features)
    X_stride = _tstride(X, order=order, mode="valid")
    X_stride = X_stride[: len(X) - lag]
    y_shift = y[order - 1 + lag :]
    return X_stride, y_shift


def _segments_to_windows(segments: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Extract contiguous window slices from segments label array.

    Returns array of `values` representing the value of each unique segment, and
    `windows` containing the `(start, stop)` indices for each contiguous segment.

    Raises an error if the segments are not contiguous.
    """
    assert segments.ndim in {1, 2}, "1d or 2d segments expected"
    is_2d = segments.ndim == 2

    values, indices = np.unique(segments, axis=0 if is_2d else None, return_index=True)
    order = np.argsort(indices)  # sort by order of appearance
    values = values[order]

    windows = []
    for val in values:
        mask = segments == val
        if is_2d:
            mask = np.all(axis=1)
        (indices,) = np.where(mask)
        assert np.max(np.diff(indices)) == 1, "expected contiguous segments"
        start, length = indices[0], len(indices)
        windows.append([start, start + length])

    windows = np.array(windows)
    return values, windows


def _tstride(
    X: np.ndarray,
    order: int = 1,
    mode: Literal["valid", "same"] = "valid",
) -> np.ndarray:
    """Select temporally offset slices of a multivariate timeseries.

    Given X of shape (n_samples, n_features), returns array of shape
    (n_samples - order + 1, order, n_features) if `mode = "valid"`, or shape
    (n_samples, order, n_features) if `mode = "same"`.

    If `mode = "same"`, the input is prepended with zeros.
    """
    if mode == "same" and order > 1:
        X = np.pad(X, [(order - 1, 0), (0, 0)])
    length = len(X) - order + 1
    assert length > 0, f"time series too short for {order=}"
    # Take slices in reverse order so that longer lags appear later.
    X_stride = np.stack(
        [X[start : start + length] for start in reversed(range(order))],
        axis=1,
    )
    return X_stride
