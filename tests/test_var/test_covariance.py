import logging

import numpy as np
import pytest

from sklearn.covariance import EmpiricalCovariance
from skarf.var._covariance import CovarianceVAR
from sklearn.utils.estimator_checks import parametrize_with_checks

from tests.conftest import Data


@pytest.mark.parametrize("degree", [1, 3])
@pytest.mark.parametrize("order", [1, 3])
@pytest.mark.parametrize("lag", [0, 1])
def test_covariance_var(random_data: Data, order: int, lag: int, degree: int):
    X, segments, sample_weight = (
        random_data.X,
        random_data.segments,
        random_data.sample_weight,
    )
    n_samples, n_features = X.shape

    random_state = np.random.RandomState(42)
    var = CovarianceVAR(
        EmpiricalCovariance(),
        order=order,
        lag=lag,
        degree=degree,
        random_state=random_state,
    )

    # Check basic fit.
    var.fit(X, segments=segments, sample_weight=sample_weight)
    assert var.coef_.shape == (order, n_features, n_features)

    # Check recovery of ground truth coefficients by sampling data from the fit model.
    if lag > 0:
        samples = var.sample(n_samples)
        var2 = CovarianceVAR(
            var.estimator_,
            order=order,
            lag=lag,
            degree=degree,
            frozen=True,
        )
        var2.fit(samples)

        score = var2.score(samples)
        assert np.isclose(score, 1.0)
        assert np.allclose(var2.coef_, var.coef_)
        assert np.allclose(var2.beta_, var.beta_)


def test_covariance_var_ridge(random_data: Data):
    # Check that the ridge penalty effectively suppresses the coefficients.
    cov = EmpiricalCovariance()
    base_model = CovarianceVAR(cov, order=2)
    ridge_model = CovarianceVAR(cov, order=2, alpha=1e5)

    base_model.fit(random_data.X)
    ridge_model.fit(random_data.X)

    base_ar_l2 = np.linalg.norm(base_model.coef_)
    ridge_ar_l2 = np.linalg.norm(ridge_model.coef_)
    logging.info(f"base l2: {base_ar_l2:.3e}, ridge l2: {ridge_ar_l2:.3e}")
    assert ridge_ar_l2 < 0.001
    assert base_ar_l2 > 0.1


@parametrize_with_checks(
    [
        CovarianceVAR(EmpiricalCovariance()),
    ],
    expected_failed_checks=lambda estimator: {
        "check_sample_weight_equivalence_on_dense_data": "binary sample weights only",
        "check_sample_weights_list": "binary sample weights only",
        "check_sample_weights_not_overwritten": "binary sample weights only",
    },
)
def test_sklearn_compatible_estimator(estimator, check):
    check(estimator)
