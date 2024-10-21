import numpy as np
import pytest


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
        sample = sample @ A.T
        samples.append(sample)
    samples = np.stack(samples)
    return A, samples


@pytest.fixture(scope="module")
def groups() -> np.ndarray:
    groups = np.concatenate(
        [np.full((100,), 3, dtype=np.int64), np.full((156,), 1, dtype=np.int64)],
    )
    return groups