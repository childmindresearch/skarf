import numpy as np
import pytest

from arfc.base import ARModel


@pytest.mark.parametrize(
    "order,lag",
    [
        (1, 1),
        (1, 2),
        (3, 1),
        (3, 2),
    ],
)
def test_tstride_tshift(random_single_data: np.ndarray, order: int, lag: int):
    model = ARModel(order=order, lag=lag)

    T, D = random_single_data.shape
    X_stride = model.tstride(random_single_data)
    X_shift = model.tshift(random_single_data)
    _stride_shift_checks(X_stride, X_shift, order, lag, T, D)


@pytest.mark.parametrize(
    "order,lag",
    [
        (1, 1),
        (1, 2),
        (3, 1),
        (3, 2),
    ],
)
def test_batch_tstride_tshift(random_batch_data: np.ndarray, order: int, lag: int):
    model = ARModel(order=order, lag=lag)

    X_stride = model.tstride(random_batch_data)
    X_shift = model.tshift(random_batch_data)

    for ii, Xi in enumerate(random_batch_data):
        T, D = Xi.shape
        _stride_shift_checks(X_stride[ii], X_shift[ii], order, lag, T, D)


def _stride_shift_checks(
    X_stride: np.ndarray,
    X_shift: np.ndarray,
    order: int,
    lag: int,
    n_tpts: int,
    dim: int,
):
    assert X_stride.shape == (order, n_tpts - order - lag + 1, dim)
    assert X_shift.shape == (n_tpts - order - lag + 1, dim)

    min_diff = np.min(np.abs(X_stride - X_shift))
    assert min_diff > 1e-6

    for ii in range(order):
        start = lag + order - 1 - ii
        assert np.allclose(X_stride[ii, start : start + 10], X_shift[:10])
