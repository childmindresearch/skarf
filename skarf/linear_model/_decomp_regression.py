import numpy as np

from sklearn.base import MetaEstimatorMixin, RegressorMixin, _fit_context, clone
from sklearn.decomposition import PCA
from sklearn.linear_model._base import LinearModel
from sklearn.utils import Bunch
from sklearn.utils.metadata_routing import (
    MetadataRouter,
    MethodMapping,
    _raise_for_params,
    _routing_enabled,
    process_routing,
)
from sklearn.utils._param_validation import HasMethods


class DecompRegression(MetaEstimatorMixin, RegressorMixin, LinearModel):
    """A regression model with a decomposition step.

    This estimator first applies a decomposition method to the input features,
    then fits a regression model on the transformed features.

    Parameters
    ----------
    decomposition : estimator
        A decomposition estimator (e.g., PCA) that implements a `fit` method.

    regression : estimator
        A regression estimator (e.g., LinearRegression) that implements a `fit` method.

    Attributes
    ----------
    decomposition_ : estimator
        The fitted decomposition estimator.

    regression_ : estimator
        The fitted regression estimator.

    coef_ : ndarray of shape (n_features,) or (n_targets, n_features)
        The estimated coefficients of the linear model.

    intercept_ : float or ndarray of shape (n_targets,)
        The intercept of the linear model.
    """

    _parameter_constraints = {
        "decomposition": [HasMethods(["fit"])],
        "regression": [HasMethods(["fit"])],
    }

    decomposition_: PCA
    """Fit decomposition estimator."""

    regression_: PCA
    """Fit linear regression estimator."""

    coef_: np.ndarray
    """Linear model coefficients of shape (n_features,) or (n_targets, n_features)."""

    intercept_: np.ndarray | float
    """Linear model intercept, either float or array of shape (n_targets,)."""

    def __init__(self, decomposition: PCA, regression: LinearModel):
        super().__init__()
        self.decomposition = decomposition
        self.regression = regression

    @_fit_context(prefer_skip_nested_validation=False)
    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        sample_weight: np.ndarray = None,
        **params,
    ):
        """Fit the decomposition and regression models.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Training data.

        y : ndarray of shape (n_samples,) or (n_samples, n_targets)
            Target values.

        sample_weight : ndarray of shape (n_samples,), default=None
            Individual weights for each sample.

        **params : dict
            Additional parameters to be passed to the fit methods of the estimators.

        Returns
        -------
        self : object
            Fitted estimator.
        """
        _raise_for_params(params, self, "fit")

        decomposition = clone(self.decomposition)
        regression = clone(self.regression)

        if _routing_enabled():
            if sample_weight is not None:
                params["sample_weight"] = sample_weight
            routed_params = process_routing(self, "fit", **params)
        else:
            routed_params = Bunch(decomposition=Bunch(fit={}), regression=Bunch(fit={}))
            if sample_weight is not None:
                routed_params.regression.fit["sample_weight"] = sample_weight

        decomposition.fit(X, **routed_params.decomposition.fit)
        XC = X @ decomposition.components_.T
        regression.fit(XC, y, **routed_params.regression.fit)

        self.decomposition_ = decomposition
        self.regression_ = regression
        self.coef_ = regression.coef_ @ decomposition.components_
        self.intercept_ = regression.intercept_
        return self

    def get_metadata_routing(self):
        """Get metadata routing of this object.

        Please check :ref:`User Guide <metadata_routing>` on how the routing
        mechanism works.

        .. versionadded:: 1.5

        Returns
        -------
        routing : MetadataRouter
            A :class:`~sklearn.utils.metadata_routing.MetadataRouter` encapsulating
            routing information.
        """
        router = (
            MetadataRouter(owner=self.__class__.__name__)
            .add_self_request(self)
            .add(
                decomposition=self.decomposition,
                method_mapping=MethodMapping().add(caller="fit", callee="fit"),
            )
            .add(
                regression=self.regression,
                method_mapping=MethodMapping().add(caller="fit", callee="fit"),
            )
        )
        return router
