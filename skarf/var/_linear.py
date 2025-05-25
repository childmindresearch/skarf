"""Regularized linear VAR models."""

from typing import Self

import numpy as np
from numpy.random import RandomState
from sklearn.base import MetaEstimatorMixin, clone, _fit_context
from sklearn.decomposition import PCA
from sklearn.linear_model._base import LinearModel
from sklearn.preprocessing import normalize
from sklearn.utils.validation import validate_data
from sklearn.utils._param_validation import HasMethods

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

    per_target : bool, default=False
        Fit separate linear model per target time series. This also excludes the target
        time series from the model.

    decomposition : estimator object, default=None
        Decomposition estimator object. Used to compute a component dictionary for
        representing linear model coefficients.

    transpose : bool, default=False
        Fit a transposed decomposition.

    random_state : int, RandomState instance, default=None
        The seed of the pseudo random number generator used when sampling.
        Note that using an int will produce identical results on each call to `sample`.
        Passing a `RandomState` instance will produce varying but reproducible sampling
        results.

    Attributes
    ----------
    coef_ : array of shape (order, n_features, n_features)
        Estimated coefficients for the VAR model. The terms are ordered by increasing
        lag.  The `i`th row of each term contains the prediction coefficients for the
        `i`th feature.

    intercept_ : array of shape (n_features,) or float
        Model intercept.

    n_features_in_ : int
        Number of features seen during :term:`fit`.

    feature_names_in_ : array of shape (`n_features_in_`,)
        Names of features seen during `fit`. Defined only when `X` has feature names
        that are all strings.

    estimator_ : Estimator object
        Fit linear model estimator. Defined only when `per_target=False`.

    estimators_ : list Estimator objects
        Fit per target linear model estimators. Defined only when `per_target=True`.

    decomposition_ : Estimator object
        Fit decomposition estimator. Defined only when `decomposition` not `None`.

    components_ : array of shape (n_components, n_features)
        Deomposition components dictionary. Defined only when `decomposition` not
        `None`.
    """

    _parameter_constraints = {
        **BaseVAR._parameter_constraints,
        "estimator": [HasMethods(["fit"])],
        "per_target": ["boolean"],
        "decomposition": [HasMethods(["fit"]), None],
        "transpose": ["boolean"],
    }

    intercept_: np.ndarray | float
    """Model intercept."""

    estimator_: LinearModel
    """Fit linear model estimator. Defined only when `per_target=False`."""

    estimators_: list[LinearModel]
    """Fit per target linear model estimators. Defined only when `per_target=True`."""

    decomposition_: PCA
    """Fit decomposition estimator. Defined only when `decomposition` not None."""

    components_: np.ndarray
    """Decomposition components dictionary. Defined when `decomposition` not None."""

    def __init__(
        self,
        estimator: LinearModel,
        order: int = 1,
        lag: int = 1,
        per_target: bool = False,
        decomposition: PCA | None = None,
        transpose: bool = False,
        random_state: int | RandomState | None = None,
    ):
        super().__init__(order=order, lag=lag, random_state=random_state)
        self.estimator = estimator
        self.per_target = per_target
        self.decomposition = decomposition
        self.transpose = transpose

    @_fit_context(prefer_skip_nested_validation=False)
    def fit(
        self,
        X: np.ndarray,
        y: None = None,
        segments: np.ndarray | None = None,
        sample_weight: np.ndarray | None = None,
        groups: np.ndarray | None = None,
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

        groups : array-like of shape (n_samples,), default=None
            Indicator array of CV groups.

        Returns
        -------
        self : object
            Returns the instance itself.
        """
        X = validate_data(self, X)

        X_stride, X_shift, _, sample_weight_shift, groups_shift = _preprocess_data(
            X,
            y=None,
            order=self.order,
            lag=self.lag,
            segments=segments,
            sample_weight=sample_weight,
            groups=groups,
        )

        params = {}
        if sample_weight_shift is not None:
            params["sample_weight"] = sample_weight_shift
        if groups_shift is not None:
            params["groups"] = groups_shift

        if self.decomposition is not None:
            decomposition = clone(self.decomposition)
            if self.transpose:
                components = decomposition.fit_transform(X.T)
                components = np.ascontiguousarray(components.T)
            else:
                decomposition.fit(X)
                components = decomposition.components_
            components = normalize(components)
        else:
            decomposition = components = None

        if self.per_target:
            estimators, coef, intercept = self._fit_linear_per_target(
                X_stride,
                X_shift,
                components=components,
                **params,
            )
        else:
            estimator, coef, intercept = self._fit_linear_joint(
                X_stride,
                X_shift,
                components=components,
                **params,
            )

        if self.per_target:
            self.estimators_ = estimators
        else:
            self.estimator_ = estimator
        self.coef_ = coef
        self.intercept_ = intercept

        if self.decomposition is not None:
            self.decomposition_ = decomposition
            self.components_ = components
        return self

    def _fit_linear_per_target(
        self,
        X_stride: np.ndarray,
        X_shift: np.ndarray,
        components: np.ndarray | None = None,
        **params,
    ):
        n_samples, _, n_features = X_stride.shape

        estimators = []
        coefs = []
        intercepts = []

        for idx in range(n_features):
            # Zero out target feature, to prevent model relying on autocorrelation
            X_stride_i = X_stride.copy()
            X_stride_i[:, :, idx] = 0

            # Project data onto compononents. Equivalently, this constrains the linear
            # regression weights to lie in the span of the components.
            if components is not None:
                X_stride_proj_i = X_stride_i @ components.T
            else:
                X_stride_proj_i = X_stride_i

            X_stride_flat_i = X_stride_proj_i.reshape(n_samples, -1)

            estimator = clone(self.estimator)
            estimator.fit(X_stride_flat_i, X_shift[:, idx], **params)

            estimators.append(estimator)
            coefs.append(estimator.coef_)
            intercepts.append(estimator.intercept_)

        coef = np.stack(coefs)
        intercept = np.array(intercepts).flatten()
        if np.allclose(intercept, 0.0):
            intercept = 0.0

        # (n_targets, order * n_features) -> (order, n_targets, n_features)
        coef = coef.reshape(n_features, self.order, -1).swapaxes(0, 1)
        if components is not None:
            coef = coef @ components

        # Fill diagonal with zero.
        coef[:, np.arange(n_features), np.arange(n_features)] = 0.0
        return estimators, coef, intercept

    def _fit_linear_joint(
        self,
        X_stride: np.ndarray,
        X_shift: np.ndarray,
        components: np.ndarray | None = None,
        **params,
    ):
        n_samples, _, n_features = X_stride.shape

        if components is not None:
            X_stride_proj = X_stride @ components.T
        else:
            X_stride_proj = X_stride

        X_stride_flat = X_stride_proj.reshape(n_samples, -1)

        estimator = clone(self.estimator)
        estimator.fit(X_stride_flat, X_shift, **params)

        coef = estimator.coef_.copy()
        intercept = estimator.intercept_

        # (n_targets, order * n_features) -> (order, n_targets, n_features)
        coef = coef.reshape(n_features, self.order, -1).swapaxes(0, 1)
        if components is not None:
            coef = coef @ components

        # Fill diagonal with zero.
        coef[:, np.arange(n_features), np.arange(n_features)] = 0.0

        # Rescale coefficients to compensate for lost diagonal.
        X_pred = np.einsum("npd,pkd->nk", X_stride, coef)
        scale = _fit_scale(X_pred, X_shift - intercept, axis=0)
        coef = scale[None, :, None] * coef
        return estimator, coef, intercept

    def _predict_strided(self, X_stride: np.ndarray) -> np.ndarray:
        X_pred = np.einsum("npd,pkd->nk", X_stride, self.coef_)
        X_pred = X_pred + self.intercept_
        return X_pred


def _fit_scale(y_pred: np.ndarray, y_true: np.ndarray, *, axis: int | None = None):
    """Re-scale predictions to fit targets."""
    scale = (y_pred * y_true).sum(axis=axis) / (y_pred * y_pred).sum(axis=axis)
    return scale
