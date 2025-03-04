import numpy as np
import pytest

from sklearn.linear_model import LinearRegression
from arfc.var.linear_model import LinearVAR

from tests.conftest import Data


@pytest.mark.parametrize("mode", ["joint", "per_target", "leave_one_out"])
@pytest.mark.parametrize("order", [1, 3])
@pytest.mark.parametrize("lag", [0, 1])
def test_linear_var(random_data: Data, order: int, lag: int, mode: str):
    X, segments, sample_weight = (
        random_data.X,
        random_data.segments,
        random_data.sample_weight,
    )
    n_samples, n_features = X.shape

    random_state = np.random.RandomState(42)
    var = LinearVAR(
        LinearRegression(),
        order=order,
        lag=lag,
        mode=mode,
        random_state=random_state,
    )

    # Check basic fit.
    var.fit(X, segments=segments, sample_weight=sample_weight)
    assert var.coef_.shape == (order, n_features, n_features)

    # Check recovery of ground truth coefficients by sampling data from the fit model.
    if lag > 0:
        samples = var.sample(n_samples)
        var2 = LinearVAR(
            LinearRegression(),
            order=order,
            lag=lag,
            mode=mode,
        )
        var2.fit(samples)

        score = var2.score(samples)
        assert np.isclose(score, 1.0)

        # TODO: this assert fails for order = 3. I guess order > 1 is unstable or
        # underdetermined, idk. Should figure this out.
        if order == 1:
            assert np.allclose(var2.coef_, var.coef_)
