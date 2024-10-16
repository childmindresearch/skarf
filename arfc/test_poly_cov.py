import numpy as np
import pytest
from sklearn.covariance import EmpiricalCovariance

from arfc.poly_cov import PolyCovARModel


@pytest.fixture(scope="module")
def rng() -> np.random.Generator:
    return np.random.default_rng(42)


@pytest.fixture(scope="module")
def random_data(rng: np.random.Generator) -> np.ndarray:
    X = rng.normal(size=(256, 64))
    return X


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
