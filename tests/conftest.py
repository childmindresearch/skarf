from typing import NamedTuple

import numpy as np
import pytest
from scipy.signal import butter, sosfiltfilt


class Data(NamedTuple):
    X: np.ndarray
    y: np.ndarray | None
    segments: np.ndarray | None
    sample_weight: np.ndarray | None
    groups: np.ndarray | None


@pytest.fixture(scope="session")
def random_data() -> Data:
    # Random X, y sampled from smooth low-dim trajectory
    rng = np.random.default_rng(42)
    n_samples, n_features, n_targets, n_components = 64, 16, 8, 2
    X_y = _random_smooth_lowdim_data(
        rng, n_samples, n_features + n_targets, n_components
    )
    X, y = X_y[:, :n_features], X_y[:, n_features:]

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


def _random_smooth_lowdim_data(
    rng: np.random.Generator,
    n_samples: int,
    n_features: int,
    n_components: int,
    sigma: float = 0.1,
) -> np.ndarray:
    U, _ = np.linalg.qr(rng.normal(size=(n_features, n_components)))
    x = rng.normal(size=(n_samples, n_components))
    x = _lowpass(x, axis=0)
    X = x @ U.T
    X = X / X.std()
    if sigma > 0:
        X = X + sigma * rng.normal(size=(n_samples, n_features))
    return X


def _lowpass(
    x: np.ndarray,
    cutoff: float = 0.08,
    fs: float = 1.0,
    axis: int = -1,
) -> np.ndarray:
    sos = butter(4, cutoff, btype="lowpass", output="sos", fs=fs)
    x = sosfiltfilt(sos, x, axis=axis, padtype="even")
    return x
