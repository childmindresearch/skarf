from itertools import repeat
from typing import Literal

import numpy as np
from sklearn.utils import check_array


def _segments_to_windows(segments: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Extract contiguous window slices from segments label array.

    Returns array of `windows` containing the `(start, stop)` indices for each
    contiguous segment, and the array of `values` for each segment.

    Raises an error if the segments are not contiguous.
    """
    assert segments.ndim in {1, 2}, "1d or 2d segments expected"
    is_2d = segments.ndim == 2

    values, indices = np.unique(segments, axis=0 if is_2d else None, return_index=True)
    order = np.argsort(indices)  # sort by order of appearance
    values = values[order]

    windows = []
    for val in values:
        mask = segments == val
        if is_2d:
            mask = np.all(mask, axis=1)
        (indices,) = np.where(mask)
        if len(indices) < 2:
            raise ValueError("Segments should have at least two time points.")
        if np.max(np.diff(indices)) > 1:
            raise ValueError("Segments should be contiguous.")
        start, length = indices[0], len(indices)
        windows.append([start, start + length])

    windows = np.array(windows)
    return windows, values


def _tstride(
    X: np.ndarray,
    order: int = 1,
    mode: Literal["valid", "same"] = "valid",
) -> np.ndarray:
    """Select temporally offset slices of a multivariate timeseries.

    Given X of shape (n_samples, n_features), returns array of shape
    (n_samples - order + 1, order, n_features) if `mode = "valid"`, or shape
    (n_samples, order, n_features) if `mode = "same"`.

    If `mode = "same"`, the input is prepended with zeros.
    """
    assert mode in {"valid", "same"}, "expected mode in {valid, same}"
    assert order > 0, "expected order > 0"

    if mode == "same" and order > 1:
        X = np.pad(X, [(order - 1, 0), (0, 0)])

    length = len(X) - order + 1
    if length <= 0:
        raise ValueError(f"Time series too short for {order=}")

    # Take slices in reverse order so that longer lags appear later.
    X_stride = np.stack(
        [X[start : start + length] for start in reversed(range(order))],
        axis=1,
    )
    return X_stride


def _optional_zip(*arrays):
    """Zip a sequence of iterables, repeating None for any that are None."""
    try:
        first = next((arr for arr in arrays if arr is not None))
    except StopIteration:
        raise ValueError("Not all arrays should be None")
    length = len(first)

    arrays = [repeat(None, length) if arr is None else arr for arr in arrays]
    yield from zip(*arrays)


def _check_segments(segments: np.ndarray, X: np.ndarray) -> np.ndarray:
    """Check validity of temporal segments array."""
    segments = check_array(segments, ensure_2d=False, input_name="segments")

    if segments.ndim not in {1, 2}:
        raise ValueError("Temporal segments must be 1D or 2D array")

    if len(segments) != len(X):
        raise ValueError("Temporal segments must be same length as X")
    return segments
