import numpy as np
import pytest

import arfc.timeseries as ts


@pytest.mark.parametrize("order", [1, 3])
def test_tstride(random_data: np.ndarray, order: int):
    T, D = random_data.shape
    data_stride = ts.tstride(random_data, order=order)
    assert data_stride.shape == (order, T - order + 1, D)


@pytest.mark.parametrize("order", [1, 3])
def test_batch_tstride(random_batch_data: np.ndarray, order: int):
    data_stride = ts.tstride(random_batch_data, order=order)

    assert len(data_stride) == len(random_batch_data)
    for ii, Xi in enumerate(random_batch_data):
        T, D = Xi.shape
        assert data_stride[ii].shape == (order, T - order + 1, D)


@pytest.mark.parametrize("lag", [1, 3])
def test_tshift(random_data: np.ndarray, lag: int):
    T, D = random_data.shape
    data_shift = ts.tshift(random_data, lag=lag)
    assert data_shift.shape == (T - lag, D)


@pytest.mark.parametrize("lag", [1, 3])
def test_batch_tshift(random_batch_data: np.ndarray, lag: int):
    data_shift = ts.tshift(random_batch_data, lag=lag)

    assert len(data_shift) == len(random_batch_data)
    for ii, Xi in enumerate(random_batch_data):
        T, D = Xi.shape
        assert data_shift[ii].shape == (T - lag, D)
