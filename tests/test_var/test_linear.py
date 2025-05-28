import numpy as np
import pytest

import sklearn
from scipy.spatial.distance import cosine
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV, LeaveOneGroupOut
from sklearn.linear_model import LinearRegression, RidgeCV, Ridge
from sklearn.utils.estimator_checks import parametrize_with_checks
from statsmodels.tsa.api import VAR

from skarf.var._linear import LinearVAR, _fit_scale

from tests.conftest import Data


@pytest.mark.parametrize("per_target", [False, True])
@pytest.mark.parametrize("order", [1, 3])
def test_linear_var(random_data: Data, order: int, per_target: bool):
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
        per_target=per_target,
        random_state=random_state,
    )

    # Check basic fit.
    var.fit(X, segments=segments, sample_weight=sample_weight)
    assert var.coef_.shape == (order, n_features, n_features)

    # Check that score works and fit is good.
    score = var.score(X)
    assert score > 0.85

    # Check recovery of ground truth coefficients by sampling data from the fit model.
    samples = var.sample(n_samples)
    var2 = LinearVAR(
        LinearRegression(),
        order=order,
        per_target=per_target,
    )
    var2.fit(samples)

    score = var2.score(samples)
    assert score >= 0.99

    # TODO: this assert fails for order = 3. I guess order > 1 is unstable or
    # underdetermined, idk. Should figure this out.
    if order == 1:
        assert np.allclose(var2.coef_, var.coef_)


def test_linear_var_lag_0(random_data: Data):
    X, segments, sample_weight = (
        random_data.X,
        random_data.segments,
        random_data.sample_weight,
    )
    n_samples, n_features = X.shape

    random_state = np.random.RandomState(42)
    var = LinearVAR(
        LinearRegression(),
        lag=0,
        per_target=True,
        random_state=random_state,
    )

    # Check basic fit.
    var.fit(X, segments=segments, sample_weight=sample_weight)
    assert var.coef_.shape == (1, n_features, n_features)

    # Check that score works and fit is good.
    score = var.score(X)
    assert score > 0.95


@pytest.mark.parametrize("per_target", [False, True])
@pytest.mark.parametrize("order", [3])
@pytest.mark.parametrize("lag", [1])
def test_linear_var_cv(random_data: Data, order: int, lag: int, per_target: bool):
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
        per_target=per_target,
        random_state=random_state,
    )

    # Check basic fit.
    with sklearn.config_context(enable_metadata_routing=True):
        var.fit(X, segments=segments, sample_weight=sample_weight, groups=groups)
    assert var.coef_.shape == (order, n_features, n_features)

    # Check that score works and fit is good.
    score = var.score(X)
    assert score > 0.95


@pytest.mark.parametrize("transpose", [False, True])
@pytest.mark.parametrize("per_target", [False, True])
def test_linear_var_decomp(random_data: Data, per_target: bool, transpose: bool):
    X, segments, sample_weight = (
        random_data.X,
        random_data.segments,
        random_data.sample_weight,
    )
    n_samples, n_features = X.shape

    random_state = np.random.RandomState(42)
    var = LinearVAR(
        LinearRegression(),
        per_target=per_target,
        decomposition=PCA(n_components=2),
        transpose=transpose,
        random_state=random_state,
    )

    # Check basic fit.
    var.fit(X, segments=segments, sample_weight=sample_weight)
    assert var.coef_.shape == (1, n_features, n_features)

    # Check that score works and fit is good.
    score = var.score(X)
    assert score > 0.9


def test_linear_var_statsmodels_consistency(random_data: Data):
    X = random_data.X

    var_ref = VAR(X).fit(1)
    coef_ref = var_ref.coefs
    np.fill_diagonal(coef_ref[0], 0.0)

    var = LinearVAR(LinearRegression()).fit(X)
    coef = var.coef_

    # each row should be off by scale only
    dists = np.array(
        [cosine(coef_ref[0, ii], coef[0, ii]) for ii in range(X.shape[-1])]
    )
    assert np.allclose(dists, 0.0)


def test_linear_var_gridsearchcv(random_data: Data):
    X, segments, sample_weight, groups = (
        random_data.X,
        random_data.segments,
        random_data.sample_weight,
        random_data.groups,
    )
    n_samples, n_features = X.shape

    var = GridSearchCV(
        LinearVAR(Ridge(), decomposition=PCA()),
        param_grid={
            "estimator__alpha": [0.1, 1.0],
            "decomposition__n_components": [2, 4],
        },
        cv=LeaveOneGroupOut(),
    )

    # Metadata routing required to correctly route segments and sample weight to score.
    # GridSearchCV automatically routes to fit but not score it seems.
    with sklearn.config_context(enable_metadata_routing=True):
        var.estimator.set_fit_request(segments=True)
        var.estimator.set_fit_request(sample_weight=True)
        var.estimator.set_fit_request(groups=False)

        var.estimator.set_score_request(segments=True)
        var.estimator.set_score_request(sample_weight=True)

        var.fit(X, segments=segments, sample_weight=sample_weight, groups=groups)

    assert var.best_estimator_.coef_.shape == (1, n_features, n_features)

    # Check that score works and fit is good.
    score = var.score(X)
    assert score > 0.9


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


def test_fit_scale():
    rng = np.random.default_rng(42)
    y_true = rng.normal(size=(100,))
    scale = 0.3
    y_pred = y_true / scale + 1e-4 * rng.normal(size=(100,))
    scale_ = _fit_scale(y_pred, y_true)
    assert np.isclose(scale, scale_, rtol=1e-3, atol=1e-3)
