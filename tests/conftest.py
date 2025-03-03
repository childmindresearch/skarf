from typing import NamedTuple

import numpy as np
import pytest


class Data(NamedTuple):
    X: np.ndarray
    y: np.ndarray | None
    segments: np.ndarray | None
    sample_weight: np.ndarray | None
    groups: np.ndarray | None


@pytest.fixture(scope="session")
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
