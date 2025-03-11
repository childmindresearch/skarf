"""Regularized linear VAR models."""

from typing import Literal, Self

import numpy as np
from numpy.random import RandomState
from sklearn.base import MetaEstimatorMixin, clone, _fit_context
from sklearn.linear_model._base import LinearModel
from sklearn.utils.validation import validate_data
from sklearn.utils._param_validation import HasMethods, StrOptions

from ._base import BaseVAR, _preprocess_data


class LinearVAR(MetaEstimatorMixin, BaseVAR):
    """Linear VAR model.

    A linear VAR model as a meta-estimator, using sklearn linear models for parameter
    fitting.

    Parameters
    ----------
    estimator : estimator object
        `LinearModel` estimator object.

    order : int, default=1
        VAR model order, i.e. the number of past "lags" to include when predicting a
        future time point.

    lag : int, default=1
        Base temporal prediction lag/offset.

    mode : {'full', 'per_target', 'leave_one_out'}, default='full'
        Model fit mode:

        * 'full' : The linear model is fit jointly for all target time series.

        * 'per_target' : A separate linear model is fit for each target. Useful for\
            doing separate cross-validation and hyperparameter tuning for each target\
            time series.

        * 'leave_one_out' : Fit a separate model for each input time series, and exclude\
            the target time series from the model.

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
        Fit linear model estimator for `mode = 'full'`.

    estimators_ : list Estimator objects
        Fit per target linear model estimators for `mode = 'per_target'`, `mode =
        'leave_one_out'`.
    """

    _parameter_constraints = {
        **BaseVAR._parameter_constraints,
        "estimator": [HasMethods(["fit"])],
        "mode": [StrOptions({"full", "per_target", "leave_one_out"})],
    }

    estimator_: LinearModel
    """Fit linear model estimator. Defined only for `mode = 'full'`."""
    estimators_: list[LinearModel]
    """Fit per target linear model estimators. Defined only for `mode != 'joint'`."""

    def __init__(
        self,
        estimator: LinearModel,
        order: int = 1,
        lag: int = 1,
        mode: Literal["full", "per_target", "leave_one_out"] = "full",
        random_state: int | RandomState | None = None,
    ):
        super().__init__(order=order, lag=lag, random_state=random_state)
        self.estimator = estimator
        self.mode = mode

    @_fit_context(prefer_skip_nested_validation=False)
    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray | None = None,
        segments: np.ndarray | None = None,
        sample_weight: np.ndarray | None = None,
        groups: np.ndarray | None = None,
    ) -> Self:
        """Fit the model with X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training multivariate time series.

        y : array-like of shape (n_samples, n_targets,) or (n_samples,) or None
            Target time series. If `None`, the data itself is used as the target.

        segments : array-like of shape (n_samples,)
            Indicator array of contiguous temporal segments in `X`.

        sample_weight : array-like of shape (n_samples,), default=None
            Sample weights. Only binary sample weights indicating time points to
            include/exclude are currently supported.

        groups : array-like of shape (n_samples,), default=None
            Indicator array of CV groups.

        Returns
        -------
        self : object
            Returns the instance itself.
        """
        if self.mode == "leave_one_out" and y is not None:
            raise ValueError(
                "'leave_one_out' mode incompatible with predicting y targets."
            )

        if y is not None:
            X, y = validate_data(self, X, y, multi_output=True)
        else:
            X = validate_data(self, X)

        X_stride, y_shift, _, sample_weight_shift, groups_shift = _preprocess_data(
            X,
            y=y,
            order=self.order,
            lag=self.lag,
            segments=segments,
            sample_weight=sample_weight,
            groups=groups,
        )
        n_features = X_stride.shape[-1]
        if y_shift.ndim == 1:
            y_shift = y_shift.reshape(-1, 1)
        n_targets = y_shift.shape[-1]

        params = {}
        if sample_weight_shift is not None:
            params["sample_weight"] = sample_weight_shift
        if groups_shift is not None:
            params["groups"] = groups_shift

        # leave one out mode, where each feature is represented as a linear combo of
        # other features not including itself, requires per-target fitting.
        per_target = self.mode in {"per_target", "leave_one_out"}
        with_diagonal = self.mode != "leave_one_out"

        if per_target:
            estimators = [
                self._fit_component(
                    X_stride, y_shift, index=ii, with_diagonal=with_diagonal, **params
                )
                for ii in range(n_targets)
            ]
            # (n_targets, order * n_features)
            coef = np.stack([estimator.coef_ for estimator in estimators])
        else:
            estimator = self._fit_joint(X_stride, y_shift, **params)
            coef = estimator.coef_

        # (n_targets, order * n_features) -> (order, n_targets, n_features)
        coef = np.ascontiguousarray(
            coef.reshape(n_targets, self.order, n_features).swapaxes(0, 1)
        )

        if not with_diagonal:
            coef[:, np.arange(n_features), np.arange(n_features)] = 0.0

        if per_target:
            self.estimators_ = estimators
        else:
            self.estimator_ = estimator
        self.coef_ = coef
        return self

    def _fit_component(
        self,
        X_stride: np.ndarray,
        y_shift: np.ndarray,
        index: int,
        with_diagonal: bool,
        **params,
    ) -> LinearModel:
        estimator = clone(self.estimator)
        if not with_diagonal:
            X_stride = X_stride.copy()
            X_stride[:, :, index] = 0
        X_stride_flat = self._flatten_strided(X_stride)
        estimator.fit(X_stride_flat, y_shift[:, index], **params)
        return estimator

    def _fit_joint(
        self,
        X_stride: np.ndarray,
        y_shift: np.ndarray,
        **params,
    ) -> LinearModel:
        estimator = clone(self.estimator)
        X_stride_flat = self._flatten_strided(X_stride)
        estimator.fit(X_stride_flat, y_shift, **params)
        return estimator

    def _predict_strided(self, X_stride: np.ndarray) -> np.ndarray:
        # predict using underlying models
        # should be equivalent to base prediction, but just to be careful
        # (one possible difference is intercept/scaling).
        per_target = self.mode in {"per_target", "leave_one_out"}
        with_diagonal = self.mode != "leave_one_out"
        if per_target:
            n_targets = self.coef_.shape[1]
            X_pred = np.stack(
                [
                    self._predict_component(
                        X_stride, index=index, with_diagonal=with_diagonal
                    )
                    for index in range(n_targets)
                ],
                axis=-1,
            )
        else:
            X_pred = self._predict_joint(X_stride)
        return X_pred

    def _predict_component(
        self,
        X_stride: np.ndarray,
        index: int,
        with_diagonal: bool,
    ) -> np.ndarray:
        if not with_diagonal:
            X_stride = X_stride.copy()
            X_stride[:, :, index] = 0
        X_stride_flat = self._flatten_strided(X_stride)
        X_pred_i = self.estimators_[index].predict(X_stride_flat)
        return X_pred_i

    def _predict_joint(self, X_stride: np.ndarray) -> np.ndarray:
        X_stride_flat = self._flatten_strided(X_stride)
        X_pred = self.estimator_.predict(X_stride_flat)
        return X_pred

    def _flatten_strided(self, X_stride: np.ndarray) -> np.ndarray:
        assert X_stride.shape[1] == self.order, "invalid strided input shape"
        N, P, D = X_stride.shape
        return X_stride.reshape((N, P * D))
