import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression

from skarf.var._multi import MultiVAR, _stack_arrays
from skarf.var._linear import LinearVAR

from tests.conftest import Data

# NOTE: not checking sklearn estimator checks for MultiVAR because of the atypical input
# shape. MultiVAR expects either a 1D array of 2D arrays, each shape (sequence_length,
# n_features), or a 3D array of shape (n_samples, sequence_length, n_features). The
# estimator checks all use a standard input X shape (n_samples, n_features), which is
# incompatible.


@pytest.mark.parametrize(
    "sample_ids_type", ["none", "array", "list", "series", "string"]
)
@pytest.mark.parametrize(
    "size_fractions",
    [
        (1.0,),
        (1.0, 1.0, 1.0),
        (0.5, 1.0, 0.8),
    ],
)
def test_multi_var(
    random_data: Data, size_fractions: tuple[float, ...], sample_ids_type: str
):
    X, segments = random_data.X, random_data.segments
    n_samples, n_features = X.shape

    X = _stack_arrays([X[: int(fraction * n_samples)] for fraction in size_fractions])
    segments = _stack_arrays(
        [segments[: int(fraction * n_samples)] for fraction in size_fractions]
    )

    match sample_ids_type:
        case "none":
            sample_ids = None
        case "array":
            sample_ids = np.arange(1, len(X) + 1)
        case "list":
            sample_ids = list(range(1, len(X) + 1))
        case "series":
            sample_ids = None
            X = pd.Series(list(X), index=range(1, len(X) + 1))
        case "string":
            sample_ids = list("abcdefg")[: len(X)]
        case _:
            raise ValueError(f"Unexpected {sample_ids_type=}")

    var = MultiVAR(LinearVAR(LinearRegression()))
    var.fit(X, segments=segments, sample_ids=sample_ids)
    assert len(var.estimators_) == len(X)

    X_pred = var.predict(X, sample_ids=sample_ids)
    assert len(X_pred) == len(X)

    score = var.score(X, sample_ids=sample_ids, segments=segments)
    assert score > 0

    X_transformed = var.transform(X, sample_ids=sample_ids)
    assert X_transformed.shape == (len(X), var.estimator.order, n_features, n_features)

    var.incremental_fit(X[:1], sample_ids=[len(X) + 2])
    assert len(var.estimators_) == len(X) + 1
