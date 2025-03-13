import numpy as np
import pytest

from skarf.datasets._samples_generator import make_spiral


@pytest.mark.parametrize(
    "init,theta,dampening",
    [
        ((1.0, 0.0), 2 * np.pi / 60, None),
        (None, None, 0.99),
    ],
)
def test_make_spirals(
    init: np.ndarray | None,
    theta: float | None,
    dampening: float | None,
):
    n_samples = 20
    X, A = make_spiral(n_samples=n_samples, init=init, theta=theta, dampening=dampening)
    assert X.shape == (n_samples, 2)
    assert A.shape == (2, 2)
    assert np.allclose(X[: n_samples - 1] @ A.T, X[1:])


def test_make_spirals_noisy():
    n_samples = 20
    X_noisy, _ = make_spiral(n_samples=n_samples, noise=0.01, random_state=42)
    X_noisy2, _ = make_spiral(n_samples=n_samples, noise=0.01, random_state=42)
    assert np.allclose(X_noisy, X_noisy2)
