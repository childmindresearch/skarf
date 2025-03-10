import numpy as np
import pytest

import sklearn
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.utils.estimator_checks import parametrize_with_checks
from skarf.var._linear import LinearVAR
from statsmodels.tsa.api import VAR

from tests.conftest import Data

# Needed for RidgeCV groups routing
sklearn.set_config(enable_metadata_routing=True)


@pytest.mark.parametrize("mode", ["full", "per_target", "leave_one_out"])
@pytest.mark.parametrize("order", [1, 3])
@pytest.mark.parametrize("lag", [0, 1])
def test_linear_var(random_data: Data, order: int, lag: int, mode: str):
    X, segments, sample_weight = (
        random_data.X,
        random_data.segments,
        random_data.sample_weight,
    )
    n_samples, n_features = X.shape

    random_state = np.random.RandomState(42)
    var = LinearVAR(
        LinearRegression(),
        order=order,
        lag=lag,
        mode=mode,
        random_state=random_state,
    )

    # Check basic fit.
    var.fit(X, segments=segments, sample_weight=sample_weight)
    assert var.coef_.shape == (order, n_features, n_features)

    # Check recovery of ground truth coefficients by sampling data from the fit model.
    if lag > 0:
        samples = var.sample(n_samples)
        var2 = LinearVAR(
            LinearRegression(),
            order=order,
            lag=lag,
            mode=mode,
        )
        var2.fit(samples)

        score = var2.score(samples)
        assert np.isclose(score, 1.0)

        # TODO: this assert fails for order = 3. I guess order > 1 is unstable or
        # underdetermined, idk. Should figure this out.
        if order == 1:
            assert np.allclose(var2.coef_, var.coef_)


@pytest.mark.parametrize("mode", ["full", "per_target"])
@pytest.mark.parametrize("order", [1, 3])
@pytest.mark.parametrize("lag", [0, 1])
def test_linear_var_with_targets(random_data: Data, order: int, lag: int, mode: str):
    X, y, segments, sample_weight = (
        random_data.X,
        random_data.y,
        random_data.segments,
        random_data.sample_weight,
    )
    n_samples, n_features = X.shape
    n_targets = y.shape[1]

    random_state = np.random.RandomState(42)
    var = LinearVAR(
        LinearRegression(),
        order=order,
        lag=lag,
        mode=mode,
        random_state=random_state,
    )

    # Check basic fit.
    var.fit(X, y=y, segments=segments, sample_weight=sample_weight)
    assert var.coef_.shape == (order, n_targets, n_features)


@pytest.mark.parametrize("mode", ["full", "per_target", "leave_one_out"])
@pytest.mark.parametrize("order", [3])
@pytest.mark.parametrize("lag", [1])
def test_linear_var_cv(random_data: Data, order: int, lag: int, mode: str):
    X, segments, sample_weight, groups = (
        random_data.X,
        random_data.segments,
        random_data.sample_weight,
        random_data.groups,
    )
    n_samples, n_features = X.shape

    random_state = np.random.RandomState(42)
    var = LinearVAR(
        RidgeCV(alphas=[0.1, 1.0, 10.0], cv=LeaveOneGroupOut()),
        order=order,
        lag=lag,
        mode=mode,
        random_state=random_state,
    )

    # Check basic fit.
    var.fit(X, segments=segments, sample_weight=sample_weight, groups=groups)
    assert var.coef_.shape == (order, n_features, n_features)
    if mode == "full":
        alphas = np.array([var.estimator_.alpha_])
    else:
        alphas = np.array([estimator.alpha_ for estimator in var.estimators_])
    assert np.all(alphas == 10.0)


def test_linear_var_statsmodels_consistency(random_data: Data):
    X = random_data.X

    var_ref = VAR(X).fit(1)
    coef_ref = var_ref.coefs

    var = LinearVAR(LinearRegression()).fit(X)
    coef = var.coef_
    assert np.allclose(coef, coef_ref)


@parametrize_with_checks(
    [
        LinearVAR(LinearRegression()),
    ],
    expected_failed_checks=lambda estimator: {
        "check_sample_weight_equivalence_on_dense_data": "binary sample weights only",
        "check_sample_weights_list": "binary sample weights only",
        "check_sample_weights_not_overwritten": "binary sample weights only",
    },
)
def test_sklearn_compatible_estimator(estimator, check):
    check(estimator)
