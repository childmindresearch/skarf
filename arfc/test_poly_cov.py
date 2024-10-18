import numpy as np
import pytest
from sklearn.covariance import EmpiricalCovariance

from arfc.poly_cov import PolyCovARModel, FrozenCovariance


@pytest.fixture(scope="module")
def rng() -> np.random.Generator:
    return np.random.default_rng(42)


@pytest.fixture(scope="module")
def random_data(rng: np.random.Generator) -> np.ndarray:
    X = rng.normal(size=(256, 64))
    return X


@pytest.fixture(scope="module")
def orth_mat_data(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    A, _ = np.linalg.qr(rng.normal(size=(64, 64)))

    sample = rng.normal(size=(64,))
    samples = [sample]
    for ii in range(1, 256):
        sample = sample @ A
        samples.append(sample)
    samples = np.stack(samples)
    return A, samples


@pytest.fixture(scope="module")
def groups() -> np.ndarray:
    groups = np.concatenate(
        [np.full((100,), 3, dtype=np.int64), np.full((156,), 1, dtype=np.int64)],
    )
    return groups


@pytest.mark.parametrize(
    "use_precision,degree,order,lag",
    [
        (False, 3, 1, 1),
        (True, 3, 1, 1),
        (False, 1, 1, 1),
        (False, 3, 4, 1),
        (False, 3, 4, 2),
    ]
)
def test_poly_cov_ar_model(
    random_data: np.ndarray,
    groups: np.ndarray,
    use_precision: bool,
    degree: int,
    order: int,
    lag: int,
):
    cov = EmpiricalCovariance()
    model = PolyCovARModel(
        cov,
        use_precision=use_precision,
        degree=degree,
        order=order,
        lag=lag,
    )

    model.fit(random_data, groups=groups)
    model.score(random_data, groups=groups)


def test_poly_cov_ar_model_ridge(
    random_data: np.ndarray,
):
    cov = EmpiricalCovariance()
    base_model = PolyCovARModel(cov, order=2)
    ridge_model = PolyCovARModel(cov, order=2, alpha=1e5)

    base_model.fit(random_data)
    ridge_model.fit(random_data)

    base_ar_l2 = np.linalg.norm(base_model.armats_)
    ridge_ar_l2 = np.linalg.norm(ridge_model.armats_)
    print(f"base l2: {base_ar_l2:.3e}, ridge l2: {ridge_ar_l2:.3e}")
    assert ridge_ar_l2 < 1e-3 < base_ar_l2


def test_poly_cov_ar_model_recovery(orth_mat_data: tuple[np.ndarray, np.ndarray]):
    A, orth_data = orth_mat_data
    cov = FrozenCovariance(A.T)
    model = PolyCovARModel(cov, with_diagonal=True, refit_cov=False)
    
    model.fit(orth_data)
    score = model.score(orth_data)
    assert np.isclose(score, 1.0)
    assert np.allclose(model.armats_[0], A)
