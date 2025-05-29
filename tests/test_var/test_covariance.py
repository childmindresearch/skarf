import numpy as np
import pytest

from sklearn.covariance import EmpiricalCovariance
from skarf.var._covariance import CovarianceVAR
from sklearn.utils.estimator_checks import parametrize_with_checks

from tests.conftest import Data


@pytest.mark.parametrize("per_target", [False, True])
@pytest.mark.parametrize("degree", [1, 3])
@pytest.mark.parametrize("order", [1, 3])
@pytest.mark.parametrize("lag", [0, 1])
def test_covariance_var(
    random_data: Data, order: int, lag: int, degree: int, per_target: bool
):
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
        per_target=per_target,
        random_state=random_state,
    )

    # Check basic fit.
    var.fit(X, segments=segments, sample_weight=sample_weight)
    assert var.coef_.shape == (order, n_features, n_features)

    # Check that score works and fit is good.
    score = var.score(X)
    assert score > (0.8 if per_target else 0.5)


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
