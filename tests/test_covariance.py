import numpy as np
import pytest
from sklearn.covariance import EmpiricalCovariance

from arfc.covariance import CovarianceARModel, FrozenCovariance


@pytest.mark.parametrize(
    "use_precision,degree,order,lag",
    [
        (False, 3, 1, 1),
        (True, 3, 1, 1),
        (False, 1, 1, 1),
        (False, 3, 4, 1),
        (False, 3, 4, 2),
    ],
)
def test_covariance_ar_model(
    random_single_data: np.ndarray,
    use_precision: bool,
    degree: int,
    order: int,
    lag: int,
):
    cov = EmpiricalCovariance()
    model = CovarianceARModel(
        cov,
        use_precision=use_precision,
        degree=degree,
        order=order,
        lag=lag,
    )

    model.fit(random_single_data)
    model.score(random_single_data)


def test_covariance_ar_model_batch(random_batch_data: np.ndarray):
    cov = EmpiricalCovariance()
    model = CovarianceARModel(cov, order=2)
    model.fit(random_batch_data)
    model.score(random_batch_data)


def test_covariance_ar_model_ridge(
    random_single_data: np.ndarray,
):
    cov = EmpiricalCovariance()
    base_model = CovarianceARModel(cov, order=2)
    ridge_model = CovarianceARModel(cov, order=2, alpha=1e5)

    base_model.fit(random_single_data)
    ridge_model.fit(random_single_data)

    base_ar_l2 = np.linalg.norm(base_model.armats_)
    ridge_ar_l2 = np.linalg.norm(ridge_model.armats_)
    print(f"base l2: {base_ar_l2:.3e}, ridge l2: {ridge_ar_l2:.3e}")
    assert ridge_ar_l2 < 1e-3 < base_ar_l2


def test_covariance_ar_model_recovery(orth_mat_data: tuple[np.ndarray, np.ndarray]):
    A, orth_data = orth_mat_data
    cov = FrozenCovariance(A)
    model = CovarianceARModel(cov, with_diagonal=True, refit_cov=False)

    model.fit(orth_data)
    score = model.score(orth_data)
    assert np.isclose(score, 1.0)
    assert np.allclose(model.armats_[0], A)
