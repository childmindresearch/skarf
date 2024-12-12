from typing import Any, Generator

import numpy as np
import pandas as pd

TimeseriesLike = list | tuple | np.ndarray | pd.Series | pd.DataFrame


def tstride(X: TimeseriesLike, order: int = 1, lag: int = 0) -> np.ndarray:
    X = as_numpy(X)

    if is_batch_timeseries(X):
        X_stride = tstack([tstride(Xi, order, lag=lag) for Xi in X])
        return X_stride

    assert is_single_timeseries(X), "input must be a single timeseries, shape (T, D)"
    assert len(X) > 2 * (order + lag), f"timeseries too short for {order=} {lag=}"

    length = len(X) - lag - order + 1
    X_stride = np.stack([X[start : start + length] for start in range(order)])
    return X_stride


def tshift(X: TimeseriesLike, lag: int = 1) -> np.ndarray:
    X = as_numpy(X)

    if is_batch_timeseries(X):
        X_shift = tstack([tshift(Xi, lag) for Xi in X])
        return X_shift

    assert is_single_timeseries(X), "input must be a single timeseries, shape (T, D)"
    assert len(X) > 2 * lag, f"timeseries too short for {lag=}"
    X_shift = X[lag:]
    return X_shift


def tstack(X: TimeseriesLike) -> np.ndarray:
    if isinstance(X, (np.ndarray, pd.DataFrame)):
        return X

    try:
        return np.stack(X)
    except ValueError:
        # https://stackoverflow.com/a/68824867
        stacked = np.empty(len(X), dtype=object)
        stacked[:] = X
        return stacked


def tflatten(X: TimeseriesLike) -> tuple[np.ndarray, np.ndarray]:
    X = as_numpy(X)

    if is_batch_timeseries(X):
        seq_ids = np.concatenate([np.full(len(Xi), ii) for ii, Xi in enumerate(X)])
        X = np.concatenate(X)
    else:
        seq_ids = np.zeros(len(X), dtype=np.int64)
    return X, seq_ids


def is_single_timeseries(X: Any) -> bool:
    return isinstance(X, np.ndarray) and X.ndim == 2


def is_batch_timeseries(X: Any) -> bool:
    if isinstance(X, pd.Series):
        X = X.values
    return isinstance(X, (list, tuple, np.ndarray)) and (
        not len(X) or is_single_timeseries(X[0])
    )


def is_grouped_timeseries(X: Any) -> bool:
    return (
        isinstance(X, pd.DataFrame)
        and len(X.columns) >= 2
        and X.dtypes.iloc[0] == np.int64
        and X.dtypes.iloc[-1] == object
        and is_batch_timeseries(X.iloc[:, -1].values)
    )


def is_timeseries(X: Any) -> bool:
    return is_single_timeseries(X) or is_batch_timeseries(X) or is_grouped_timeseries(X)


def iter_groups(
    X: TimeseriesLike,
) -> Generator[tuple[int, TimeseriesLike], None, None]:
    if is_single_timeseries(X):
        yield 0, X
    elif is_batch_timeseries(X):
        yield from enumerate(X)
    else:
        assert is_grouped_timeseries(X), "Invalid X"
        for group, df in X.groupby(X.columns[0]):
            # TODO: there may be a use case for returning a dataframe, if we want to
            # support multi-level grouped time series, with e.g. nested subject, session
            # group columns
            yield group, df.iloc[:, -1].values


def stack_groups(
    grouped: list[tuple[int, TimeseriesLike]],
    column_name: str = "group",
) -> pd.DataFrame:
    dfs = []
    for group, df in grouped:
        df = as_pandas(df).copy()
        df.insert(0, column_name, group)
        dfs.append(df)
    df = pd.concat(dfs, ignore_index=True)
    return df


def as_numpy(X: TimeseriesLike) -> np.ndarray:
    match type(X):
        case np.ndarray:
            return X
        case pd.Series:
            return X.values
        case pd.DataFrame:
            assert is_grouped_timeseries(
                X
            ), "expected grouped timeseries if input is a dataframe"
            return X.iloc[:, -1].values
        case _:
            return tstack(X)


def as_pandas(X: TimeseriesLike) -> pd.DataFrame:
    match type(X):
        case pd.DataFrame:
            return X
        case np.ndarray:
            if is_single_timeseries(X):
                X = [X]
            elif is_batch_timeseries(X):
                X = list(X)
            return pd.DataFrame({"timeseries": X})
        case _:
            return pd.DataFrame({"timeseries": X})
