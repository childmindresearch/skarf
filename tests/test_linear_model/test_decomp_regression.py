import numpy as np
import pytest
import sklearn
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.utils.estimator_checks import parametrize_with_checks

from skarf.linear_model._decomp_regression import DecompRegression


@pytest.mark.parametrize("transpose", [False, True])
@pytest.mark.parametrize("routing_enabled", [False, True])
def test_decomp_regression(routing_enabled: bool, transpose: bool):
    sklearn.set_config(enable_metadata_routing=routing_enabled)

    rng = np.random.default_rng(42)
    n_samples, n_features, n_targets = 40, 10, 4
    n_components = 2
    X = rng.normal(size=(n_samples, n_features))
    y = rng.normal(size=(n_samples, n_targets))

    # Check that fit works and score is reasonable.
    decomposition = PCA(n_components=n_components)
    regression = Ridge()
    if routing_enabled:
        regression.set_fit_request(sample_weight=True)

    model = DecompRegression(decomposition, regression, transpose=transpose)
    model.fit(X, y)
    score = model.score(X, y)
    assert score > 0

    # Check that sample weight has an effect.
    coef = model.coef_
    sample_weight = rng.random(size=(n_samples,))
    model.fit(X, y, sample_weight=sample_weight)
    assert not np.allclose(coef, model.coef_)

    # Check that extra params are routed correctly to the sub-estimator.
    coef = model.coef_
    groups = np.concatenate([np.zeros(n_samples // 2), np.ones(n_samples // 2)])
    regression = RidgeCV(cv=LeaveOneGroupOut())
    if routing_enabled:
        regression.set_fit_request(sample_weight=True)
    model = DecompRegression(decomposition, regression, transpose=transpose)

    if not routing_enabled:
        with pytest.raises(ValueError):
            model.fit(X, y, sample_weight=sample_weight, groups=groups)
    else:
        model.fit(X, y, sample_weight=sample_weight, groups=groups)
        assert not np.allclose(coef, model.coef_)


@parametrize_with_checks(
    [
        DecompRegression(PCA(), Ridge()),
    ],
)
def test_sklearn_compatible_estimator(estimator, check):
    check(estimator)
