import numpy as np
import pandas as pd
import pytest
from pytest import FixtureRequest


@pytest.fixture(scope="module")
def rng() -> np.random.Generator:
    return np.random.default_rng(42)


@pytest.fixture(scope="module")
def random_single_data(rng: np.random.Generator) -> np.ndarray:
    X = rng.normal(size=(256, 64))
    return X


@pytest.fixture(
    scope="module",
    params=[[128, 256, 192], [128, 128, 128]],
)
def random_batch_data(request: FixtureRequest, rng: np.random.Generator) -> np.ndarray:
    sizes = request.param
    X = [rng.normal(size=(size, 64)) for size in sizes]
    X = np.stack(X) if len(set(sizes)) == 1 else np.array(X, dtype=object)
    return X


@pytest.fixture(scope="module")
def random_group_data(rng: np.random.Generator) -> pd.DataFrame:
    groups = [1, 1, 2, 2, 3]
    sizes = [128, 256, 192, 64, 128]
    X = pd.DataFrame.from_records(
        [
            {"group": group, "timeseries": rng.normal(size=(size, 64))}
            for group, size in zip(groups, sizes)
        ]
    )
    return X


@pytest.fixture(scope="module", params=["single", "batch", "group"])
def random_data(
    request: FixtureRequest,
    random_single_data: np.ndarray,
    random_batch_data: np.ndarray,
    random_group_data: pd.DataFrame,
) -> np.ndarray | pd.DataFrame:
    data_map = {
        "single": random_single_data,
        "batch": random_batch_data,
        "group": random_group_data,
    }
    return data_map[request.param]


@pytest.fixture(scope="module")
def orth_mat_data(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    A, _ = np.linalg.qr(rng.normal(size=(64, 64)))

    sample = rng.normal(size=(64,))
    samples = [sample]
    for ii in range(1, 256):
        sample = sample @ A.T
        samples.append(sample)
    samples = np.stack(samples)
    return A, samples
