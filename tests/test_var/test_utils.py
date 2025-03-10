import numpy as np
import pytest

import skarf.var._utils as ut


def test_segments_to_windows():
    # Create contiguous segments of different lengths.
    lengths = [24, 10, 16]
    segment_values = [3, 2, 5]
    segments = np.concatenate(
        [np.full(length, value) for length, value in zip(lengths, segment_values)]
    )

    windows, values = ut._segments_to_windows(segments)

    # Check that windows match expected.
    stops = np.cumsum(lengths)
    starts = stops - np.array(lengths)
    expected_windows = np.stack([starts, stops], axis=1)
    assert np.array_equal(windows, expected_windows)

    # Check that values match expected.
    assert np.array_equal(values, np.array(segment_values))


def test_segments_to_windows_2d():
    # Create contiguous segments of different lengths.
    # Use two columns of segment values, simulating e.g. session/run.
    lengths = [8, 10, 16, 12]
    indices = np.concatenate([np.full(length, ii) for ii, length in enumerate(lengths)])
    segment_values = np.array(
        [
            [1, 0],
            [0, 1],
            [1, 1],
            [0, 0],
        ]
    )
    segments = segment_values[indices]

    _, values = ut._segments_to_windows(segments)
    assert np.array_equal(values, segment_values)


def test_segments_to_windows_noncontiguous():
    # Create contiguous segments of different lengths.
    # Repeate a value to represent a discontiguous segment.
    lengths = [24, 10, 16]
    segment_values = [1, 0, 1]
    segments = np.concatenate(
        [np.full(length, value) for length, value in zip(lengths, segment_values)]
    )

    with pytest.raises(ValueError):
        ut._segments_to_windows(segments)


def test_segments_to_windows_short():
    # Create contiguous segments of different lengths.
    # Make one segment too short.
    lengths = [24, 1, 16]
    segment_values = [3, 2, 5]
    segments = np.concatenate(
        [np.full(length, value) for length, value in zip(lengths, segment_values)]
    )

    with pytest.raises(ValueError):
        ut._segments_to_windows(segments)


@pytest.mark.parametrize("mode", ["valid", "same"])
@pytest.mark.parametrize("order", [1, 2, 3, 8])
def test_tstride(mode: str, order: int):
    # Random time series.
    rng = np.random.default_rng(42)
    n_samples, n_features = 64, 16
    X = rng.normal(size=(n_samples, n_features))

    # Check expected shape, (n_samples, order, n_features)
    X_stride = ut._tstride(X, order=order, mode=mode)
    expected_length = n_samples if mode == "same" else n_samples - order + 1
    assert X_stride.shape == (expected_length, order, n_features)

    # Check correct striding at an arbitrary middle index
    idx = 23
    X_slice = X_stride[idx, :, 0]
    pad = order - 1 if mode == "same" else 0
    expected_X_slice = X[idx + np.arange(order)[::-1] - pad, 0]
    assert np.array_equal(X_slice, expected_X_slice)

    # Check zero padding.
    if mode == "same" and order > 1:
        assert np.all(X_stride[: order - 1, order - 1] == 0)
        assert ~np.any(X_stride[: order - 1, 0] == 0)


def test_tstride_short():
    # Random short time series.
    rng = np.random.default_rng(42)
    n_samples, n_features = 4, 16
    X = rng.normal(size=(n_samples, n_features))

    with pytest.raises(ValueError):
        ut._tstride(X, order=8)


def test_optional_zip():
    # Check that optional zip indeed repeats None values.
    count = 0
    for a, b, c in ut._optional_zip(range(4), None, range(3)):
        count += 1
        assert b is None

    # Nb that zip does not enforce equal length and drops extra values.
    assert count == 3


def test_optional_zip_all_none():
    with pytest.raises(ValueError):
        list(ut._optional_zip(None, None, None))
