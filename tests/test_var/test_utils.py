import numpy as np
import pytest

import arfc.var._utils as ut


def test_segments_to_windows_basic():
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
