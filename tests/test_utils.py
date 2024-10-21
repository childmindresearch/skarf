import numpy as np
import pytest

import arfc.utils as ut


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
    "order,lag",
    [
        (1, 1),
        (1, 4),
        (3, 1),
        (3, 4),
    ]
)
def test_tsplit(random_data: np.ndarray, order: int, lag: int):
    data_pres, data_post = ut.tsplit(random_data, order=order, lag=lag)

    assert data_post.ndim == 2
    T, D = data_post.shape
    assert data_pres.shape == (T, order, D)

    dist = np.min(np.abs(data_pres - data_post[:, None, :]))
    assert not np.isclose(dist, 0.0)


@pytest.mark.parametrize(
    "order,lag",
    [
        (1, 1),
        (1, 4),
        (3, 1),
        (3, 4),
    ]
)
def test_group_tsplit(
    random_data: np.ndarray, groups: np.ndarray, order: int, lag: int,
):
    data_pres, data_post, split_groups = ut.group_tsplit(
        random_data, groups, order=order, lag=lag
    )

    assert data_post.ndim == 2
    T, D = data_post.shape
    assert data_pres.shape == (T, order, D)

    dist = np.min(np.abs(data_pres - data_post[:, None, :]))
    assert not np.isclose(dist, 0.0)

    uniq, index = np.unique(split_groups, return_index=True)
    assert np.array_equal(uniq, [1, 3])
    assert index[0] > index[1]
