from typing import Any

import numpy as np
import pandas as pd

ArrayLike = list | tuple | np.ndarray | pd.Series


def tstride(X: ArrayLike, order: int = 1, lag: int = 0) -> np.ndarray:
    if is_batch_timeseries(X):
        X_stride = tstack([tstride(Xi, order, lag=lag) for Xi in X])
        return X_stride

    assert is_single_timeseries(X), "input must be a single timeseries, shape (T, D)"
    assert len(X) > 2 * (order + lag), f"timeseries too short for {order=} {lag=}"

    length = len(X) - lag - order + 1
    X_stride = np.stack([X[start : start + length] for start in range(order)])
    return X_stride


def tshift(X: ArrayLike, lag: int = 1) -> np.ndarray:
    if is_batch_timeseries(X):
        X_shift = tstack([tshift(Xi, lag) for Xi in X])
        return X_shift

    assert is_single_timeseries(X), "input must be a single timeseries, shape (T, D)"
    assert len(X) > 2 * lag, f"timeseries too short for {lag=}"
    X_shift = X[lag:]
    return X_shift


def tstack(X: ArrayLike) -> np.ndarray:
    try:
        return np.stack(X)
    except ValueError:
        # https://stackoverflow.com/a/68824867
        stacked = np.empty(len(X), dtype=object)
        stacked[:] = X
        return stacked


def is_single_timeseries(X: Any) -> bool:
    return isinstance(X, np.ndarray) and X.ndim == 2


def is_batch_timeseries(X: Any) -> bool:
    return isinstance(X, ArrayLike) and (not len(X) or is_single_timeseries(X[0]))


def is_timeseries(X: Any) -> bool:
    return is_single_timeseries(X) or is_batch_timeseries(X)
