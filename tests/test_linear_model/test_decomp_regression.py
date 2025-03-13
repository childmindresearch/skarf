import numpy as np
import pytest
import sklearn
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge, RidgeCV
from skarf.linear_model._decomp_regression import DecompRegression
from sklearn.model_selection import LeaveOneGroupOut


@pytest.mark.parametrize("routing_enabled", [False, True])
def test_decomp_regression(routing_enabled: bool):
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

    model = DecompRegression(decomposition, regression)
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
    model = DecompRegression(decomposition, regression)

    if not routing_enabled:
        with pytest.raises(ValueError):
            model.fit(X, y, sample_weight=sample_weight, groups=groups)
    else:
        model.fit(X, y, sample_weight=sample_weight, groups=groups)
        assert not np.allclose(coef, model.coef_)
