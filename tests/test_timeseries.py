import numpy as np
import pandas as pd
import pytest

import arfc.timeseries as ts


@pytest.mark.parametrize("order", [1, 3])
def test_tstride(random_data: np.ndarray | pd.DataFrame, order: int):
    data_stride = ts.tstride(random_data, order=order)

    if ts.is_single_timeseries(random_data):
        T, D = random_data.shape
        assert data_stride.shape == (order, T - order + 1, D)
    else:
        random_data = ts.as_numpy(random_data)  # drop group columns
        for ii, Xi in ts.iter_groups(random_data):
            T, D = Xi.shape
            assert data_stride[ii].shape == (order, T - order + 1, D)


@pytest.mark.parametrize("lag", [1, 3])
def test_tshift(random_data: np.ndarray | pd.DataFrame, lag: int):
    data_shift = ts.tshift(random_data, lag=lag)

    if ts.is_single_timeseries(random_data):
        T, D = random_data.shape
        assert data_shift.shape == (T - lag, D)
    else:
        random_data = ts.as_numpy(random_data)  # drop group columns
        for ii, Xi in ts.iter_groups(random_data):
            T, D = Xi.shape
            assert data_shift[ii].shape == (T - lag, D)
