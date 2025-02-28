from typing import NamedTuple

import numpy as np
import pytest

from arfc.var._base import _align_X_y, _preprocess_data


class Data(NamedTuple):
    X: np.ndarray
    y: np.ndarray
    segments: np.ndarray
    sample_weight: np.ndarray
    groups: np.ndarray


@pytest.fixture()
def random_data() -> Data:
    # Random X, y
    rng = np.random.default_rng(42)
    n_samples, n_features, n_targets = 64, 16, 8
    X = rng.normal(size=(n_samples, n_features))
    y = rng.normal(size=(n_samples, n_targets))

    # Arbitrary segments
    lengths = [16, 16, 16, 16]
    segment_values = [3, 2, 5, 1]
    segments = np.concatenate(
        [np.full(length, value) for length, value in zip(lengths, segment_values)]
    )

    # Drop random time points
    sample_weight = np.ones(len(X))
    sample_weight[[12, 23, 41, 59]] = 0.0

    # Arbitrary CV groups
    groups = np.concatenate([np.zeros(32, dtype=np.int64), np.ones(32, dtype=np.int64)])
    return Data(X, y, segments, sample_weight, groups)


@pytest.mark.parametrize("order", [1, 2, 3, 8])
@pytest.mark.parametrize("lag", [0, 1, 2])
def test_align_X_y(random_data: Data, order: int, lag: int):
    X, y = random_data.X, random_data.y
    n_samples, n_features = X.shape
    n_targets = y.shape[1]

    # Check expected shape.
    X_stride, y_shift = _align_X_y(X, y, order=order, lag=lag)
    expected_length = n_samples - order - lag + 1
    assert X_stride.shape == (expected_length, order, n_features)
    assert y_shift.shape == (expected_length, n_targets)

    # Check correct striding at an arbitrary middle index
    idx = 23
    X_slice = X_stride[idx, :, 0]
    expected_X_slice = X[idx + np.arange(order)[::-1], 0]
    assert np.array_equal(X_slice, expected_X_slice)

    y_slice = y_shift[idx, :4]
    expected_y_slice = y[idx + order + lag - 1, :4]
    assert np.array_equal(y_slice, expected_y_slice)


@pytest.mark.parametrize("order", [1, 3])
@pytest.mark.parametrize("lag", [0, 1])
def test_preprocess_data(random_data: Data, order: int, lag: int):
    X, y, segments, sample_weight, groups = random_data
    n_samples, n_features = X.shape
    n_targets = y.shape[1]
    n_segments = len(np.unique(segments))
    n_drop_samples = np.sum(sample_weight == 0)

    preproc_data = _preprocess_data(
        X,
        y,
        order=order,
        lag=lag,
        segments=segments,
        sample_weight=sample_weight,
        groups=groups,
    )

    X_stride, y_shift, segments_shift, sample_weight_shift, groups_shift = preproc_data

    # Each segment is truncated independently.
    expected_length = n_samples - n_segments * (order + lag - 1)
    assert X_stride.shape == (expected_length, order, n_features)
    assert y_shift.shape == (expected_length, n_targets)
    assert (
        segments_shift.shape
        == sample_weight_shift.shape
        == groups_shift.shape
        == (expected_length,)
    )

    # Check that each dropped time point is expanded.
    assert np.sum(sample_weight_shift == 0) == (order + (lag > 0)) * n_drop_samples
