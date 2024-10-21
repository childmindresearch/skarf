import numpy as np
import pytest
from sklearn.linear_model import LinearRegression

from arfc.linear_model import LinearARModel


@pytest.mark.parametrize(
    "order,lag,with_diagonal,per_target",
    [
        (3, 1, True, False),
        (3, 2, True, False),
        (3, 1, True, True),
        (3, 1, False, True),
    ],
)
def test_linear_ar_model(
    random_data: np.ndarray,
    groups: np.ndarray,
    order: int,
    lag: int,
    with_diagonal: bool,
    per_target: bool,
):
    lin = LinearRegression(fit_intercept=False)
    model = LinearARModel(
        lin,
        order=order,
        lag=lag,
        with_diagonal=with_diagonal,
        per_target=per_target,
    )

    model.fit(random_data, groups=groups)
    model.score(random_data, groups=groups)


@pytest.mark.parametrize(
    "with_diagonal,per_target",
    [
        (True, False),
        (True, True),
        (False, True),
    ],
)
def test_linear_ar_model_predict(
    random_data: np.ndarray,
    with_diagonal: bool,
    per_target: bool,
):
    lin = LinearRegression(fit_intercept=False)
    model = LinearARModel(
        lin,
        order=3,
        lag=1,
        with_diagonal=with_diagonal,
        per_target=per_target,
    )

    model.fit(random_data)
    X_pres, _, _ = model.tsplit(random_data)

    X_pred_base = super(LinearARModel, model).predict(X_pres)
    X_pred = model.predict(X_pres)
    assert np.allclose(X_pred, X_pred_base)


def test_linear_ar_model_recovery(orth_mat_data: tuple[np.ndarray, np.ndarray]):
    A, orth_data = orth_mat_data
    lin = LinearRegression(fit_intercept=False)
    model = LinearARModel(lin)

    model.fit(orth_data)
    score = model.score(orth_data)
    assert np.isclose(score, 1.0)
    assert np.allclose(model.armats_[0], A)
