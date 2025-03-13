"""Toy synthetic VAR datasets."""

from numbers import Integral, Real

import numpy as np
from numpy.random import RandomState
from sklearn.utils.validation import check_array, check_random_state
from sklearn.utils._param_validation import Interval, validate_params

DEFAULT_THETA = 2 * np.pi / 60


@validate_params(
    {
        "n_samples": [Interval(Integral, 1, None, closed="left")],
        "init": ["array-like", None],
        "theta": [Interval(Real, 0, None, closed="neither"), None],
        "dampening": [Interval(Real, 0, None, closed="neither"), None],
        "noise": [Interval(Real, 0, None, closed="left"), None],
        "random_state": ["random_state"],
    },
    prefer_skip_nested_validation=True,
)
def make_spiral(
    n_samples: int,
    init: np.ndarray | None = (1.0, 0.0),
    theta: float | None = DEFAULT_THETA,
    dampening: float | None = None,
    noise: float | None = None,
    random_state: int | RandomState | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate a spiral time series using a VAR(1) process.

    Parameters
    ----------
    n_samples : int
        Number of time steps in the generated time series.

    init : array-like of shape (2,) or None, default=(1.0, 0.0)
        Initial value. If None, a random point on the unit circle is used.

    theta : float or None, default=2 * np.pi / 60
        Rotation angle (in radians) per step. If None, a value is sampled uniformly
        between `2 * np.pi / 120` and `2 * np.pi / 60`.

    dampening : float, default=None
        Spiral dampening term. Dynamics converge to 0 if less than 1, and blow up if
        greater than 1.

    noise : float, default=None
        Standard deviation of Gaussian noise added to the process. If None, no noise is
        added.

    random_state : int, RandomState, default=None
        Random seed or RandomState instance for reproducibility.

    Returns
    -------
    X : ndarray of shape (n_samples, 2)
        The generated time series.

    A : ndarray of shape (2, 2)
        The VAR transition matrix (i.e. rotation matrix).
    """
    rng = check_random_state(random_state)

    if init is None:
        # Sample random points on the unit circle.
        init = rng.randn(2)
        init /= np.linalg.norm(init)
    else:
        # Fixed init.
        init = check_array(init, ensure_2d=False)
        if init.shape != (2,):
            raise ValueError("Invalid init, expected 1D array of length 2.")

    if theta is None:
        # Random rates for each pair.
        theta = rng.uniform(DEFAULT_THETA / 2, DEFAULT_THETA)

    A = _rotation_matrix(theta)
    if dampening is not None:
        A = A * dampening
    X = _generate_var1(A, n_samples, x_init=init, noise=noise, random_state=rng)
    return X, A


def _rotation_matrix(theta: float) -> np.ndarray:
    """Construct 2D rotation matrix by angle theta."""
    return np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])


def _generate_var1(
    A: np.ndarray,
    n_samples: int = 100,
    x_init: np.ndarray | None = None,
    noise: float | None = None,
    random_state: int | RandomState | None = None,
) -> np.ndarray:
    """Generate data according to a noisy VAR(1) process."""
    A = check_array(A)
    if A.shape[0] != A.shape[1]:
        raise ValueError("Invalid A, expected square matrix.")
    n_features = A.shape[1]

    rng = check_random_state(random_state)

    if x_init is None:
        x_init = rng.randn(n_features)
    else:
        x_init = check_array(x_init, ensure_2d=False)
    if x_init.shape != (n_features,):
        raise ValueError(f"Invalid x_init, expected shape={(n_features,)}.")

    X = np.zeros((n_samples, n_features))
    X[0] = x_init
    for ii in range(1, n_samples):
        x_i = X[ii - 1] @ A.T
        if noise is not None and noise > 0:
            x_i = x_i + noise * rng.randn(n_features)
        X[ii] = x_i

    return X
