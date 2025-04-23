import numpy as np
import pytest

from skarf.var._base import BaseVAR, _align_X_y, _preprocess_data

from tests.conftest import Data


class DummyVAR(BaseVAR):
    def fit(self, X, y=None, segments=None, sample_weight=None, **params):
        self.coef_ = X


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

    # Try with no extra arguments.
    preproc_data2 = _preprocess_data(X, order=order, lag=lag)
    assert preproc_data2.segments_shift is None
    assert preproc_data2.sample_weight_shift is None
    assert preproc_data2.groups_shift is None


@pytest.mark.parametrize("order", [1, 3])
@pytest.mark.parametrize("lag", [0, 1])
def test_base_var(random_data: Data, order: int, lag: int):
    X, segments, sample_weight = (
        random_data.X,
        random_data.segments,
        random_data.sample_weight,
    )
    n_samples, n_features = X.shape

    random_state = np.random.RandomState(42)
    var = DummyVAR(order=order, lag=lag, random_state=random_state)

    # Random orthonormal basis as coefficients.
    A = random_state.randn(order, n_features, n_features)
    Q, _ = np.linalg.qr(A)
    var.fit(Q.swapaxes(1, 2) / order**0.5)

    assert var.coef_.shape == (order, n_features, n_features)

    # Check default predict. Note the input is zero padded at the front so that the
    # output matches the input.
    X = random_data.X
    X_pred = var.predict(X)
    assert X_pred.shape == (n_samples, n_features)

    # Check scoring works. Ofc the model should not fit the data well at all.
    score = var.score(X, segments=segments, sample_weight=sample_weight)
    assert isinstance(score, float)

    # Check sampling. Note that for order 1, the samples just orbit around on the unit
    # sphere more or less. For order 3 it's a little more complicated, but with the
    # scaling by 1 / sqrt(order), they seem to stay away from zero (?).
    if lag > 0:
        X_sample = var.sample(n_samples)
        assert X_sample.shape == (n_samples, n_features)
        assert np.abs(np.mean(X_sample)) < 0.1
        assert np.std(X_sample) > 0.5
    else:
        with pytest.raises(RuntimeError):
            var.sample(n_samples)
