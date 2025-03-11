"""Fit a separate VAR model for each sample (e.g. subject) in a dataset."""

import numbers
from typing import Any, Self

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.base import (
    BaseEstimator,
    MetaEstimatorMixin,
    TransformerMixin,
    clone,
    _fit_context,
)
from sklearn.utils.validation import check_is_fitted, check_array, validate_data
from sklearn.utils.parallel import Parallel, delayed
from sklearn.utils._param_validation import HasMethods

from ._base import BaseVAR
from ._utils import _optional_zip


class MultiVAR(TransformerMixin, MetaEstimatorMixin, BaseEstimator):
    """Multi-sample VAR model.

    Fit a separate VAR model for each sample (e.g. subject) in a dataset.

    Parameters
    ----------
    estimator : estimator object
        VAR estimator object.

    n_jobs : int, default=None
        Number of jobs to run in parallel.
        ``None`` means 1 unless in a ``joblib.parallel_backend`` context.
        ``-1`` means using all processors.

    Attributes
    ----------
    estimators_ : dict of estimators
        Dict mapping sample IDs to fit VAR estimators.

    n_features_in_ : int
        Number of features seen during :term:`fit`.
    """

    _parameter_constraints = {
        "estimator": [HasMethods(["fit"])],
        "n_jobs": [numbers.Integral, None],
    }

    estimators_: dict[int, BaseVAR]
    """Mapping of sample IDs to fit VAR estimators."""

    n_features_in_: int
    """Number of features seen during `fit`."""

    def __init__(self, estimator: BaseVAR, n_jobs: int | None = None):
        self.estimator = estimator
        self.n_jobs = n_jobs

    @_fit_context(prefer_skip_nested_validation=False)
    def fit(
        self,
        X: np.ndarray | pd.Series,
        y: None = None,
        sample_ids: np.ndarray | None = None,
        **params,
    ) -> Self:
        """Fit the model with X.

        Parameters
        ----------
        X : array-like of shape (n_samples, sequence_length, n_features)
            Array of multiple training multivariate time series. If the sequences are
            different length, ``X`` should be a 1D ndarray of dtype object, where each
            element is a 2D array. ``X`` may also be a pandas ``Series`` object, with
            its index encoding each sample ID (e.g. subject ID).

        y : Ignored
            Ignored

        sample_ids : array of shape (n_samples,), default=None
            Array of sample IDs (e.g. subject ID) for each training time series. If
            ``None``, the sample IDs will be extracted from the index of ``X`` in case
            ``X`` is a ``Series``. Otherwise, sample IDs ``0, ..., n_samples - 1`` are
            assumed.

        **params : dict of str -> object
            Parameters to pass through to the underlying VAR ``fit()`` method.

        Returns
        -------
        self : object
            Returns the instance itself.
        """
        X, sample_ids = _check_X_sample_ids(X, sample_ids, unique=True)
        # Check the first sample time series, sets n_features_in_
        validate_data(self, X[0])
        self.estimators_ = self._batch_fit(X, sample_ids=sample_ids, **params)
        return self

    def incremental_fit(
        self,
        X: np.ndarray | pd.Series,
        y: None = None,
        sample_ids: np.ndarray | None = None,
        **params,
    ) -> Self:
        """Partial model fit for new samples.

        Parameters
        ----------
        X : array-like of shape (n_samples, sequence_length, n_features)
            Array of multiple multivariate time series. If the sequences are different
            length, ``X`` should be a 1D ndarray of dtype object, where each element is
            a 2D array. ``X`` may also be a pandas ``Series`` object, with its index
            encoding each sample ID (e.g. subject ID). Should contain new sample time
            series not in training data.

        y : Ignored
            Ignored

        sample_ids : array of shape (n_samples,), default=None
            Array of sample IDs (e.g. subject ID) for each training time series. If
            ``None``, the sample IDs will be extracted from the index of ``X`` in case
            ``X`` is a ``Series``. Otherwise, sample IDs ``0, ..., n_samples - 1`` are
            assumed. Should contain new sample IDs not in training data.

        **params : dict of str -> object
            Parameters to pass through to the underlying VAR ``fit()`` method.

        Returns
        -------
        self : object
            Returns the instance itself.
        """
        check_is_fitted(self)
        X, sample_ids = _check_X_sample_ids(X, sample_ids, unique=True)
        validate_data(self, X[0], reset=False)

        for sample_id in sample_ids:
            if sample_id in self.estimators_:
                raise ValueError(f"Sample ID {sample_id} already fit.")

        estimators = self._batch_fit(X, sample_ids=sample_ids, **params)
        self.estimators_.update(estimators)
        return self

    def _batch_fit(
        self,
        X: np.ndarray,
        sample_ids: np.ndarray,
        **params,
    ) -> dict[int, BaseVAR]:
        params_values = list(params.values())
        jobs = []
        for X_i, *params_values_i in _optional_zip(X, *params_values):
            params_i = {k: v for k, v in zip(params, params_values_i) if v is not None}
            jobs.append(delayed(self._fit_single)(X=X_i, **params_i))
        results = Parallel(n_jobs=self.n_jobs)(jobs)
        estimators = dict(zip(sample_ids, results))
        return estimators

    def _fit_single(self, X: np.ndarray, **params) -> BaseVAR:
        estimator = clone(self.estimator)
        estimator.fit(X, **params)
        return estimator

    def predict(
        self,
        X: np.ndarray | pd.Series,
        sample_ids: np.ndarray | None = None,
    ) -> np.ndarray:
        """Predict time series values for next time steps.

        Parameters
        ----------
        X : array-like of shape (n_samples, sequence_length, n_features)
            Array of multiple multivariate time series. If the sequences are different
            length, ``X`` should be a 1D ndarray of dtype object, where each element is
            a 2D array. ``X`` may also be a pandas ``Series`` object, with its index
            encoding each sample ID (e.g. subject ID).

        sample_ids : array of shape (n_samples,), default=None
            Array of sample IDs (e.g. subject ID) for each training time series. If
            ``None``, the sample IDs will be extracted from the index of ``X`` in case
            ``X`` is a ``Series``. Otherwise, sample IDs ``0, ..., n_samples - 1`` are
            assumed.

        Returns
        -------
        X_pred : array-like of shape (n_samples, sequence_length, n_features)
            Next time step predictions for each input time series in ``X``. For each
            time series, the estimator for the corresponding sample ID is used for
            prediction.
        """
        check_is_fitted(self)
        X, sample_ids = _check_X_sample_ids(X, sample_ids)
        validate_data(self, X[0], reset=False)

        X_pred = []
        for sample_id, X_i in zip(sample_ids, X):
            X_pred_i = self.estimators_[sample_id].predict(X_i)
            X_pred.append(X_pred_i)
        X_pred = _stack_arrays(X_pred)
        return X_pred

    def score(
        self,
        X: np.ndarray | pd.Series,
        y: None = None,
        sample_ids: np.ndarray | None = None,
        **params,
    ) -> float:
        """Return the prediction score for the model (by default R2).

        Parameters
        ----------
        X : array-like of shape (n_samples, sequence_length, n_features)
            Array of multiple multivariate time series. If the sequences are different
            length, ``X`` should be a 1D ndarray of dtype object, where each element is
            a 2D array. ``X`` may also be a pandas ``Series`` object, with its index
            encoding each sample ID (e.g. subject ID).

        y : Ignored
            Ignored

        sample_ids : array of shape (n_samples,), default=None
            Array of sample IDs (e.g. subject ID) for each training time series. If
            ``None``, the sample IDs will be extracted from the index of ``X`` in case
            ``X`` is a ``Series``. Otherwise, sample IDs ``0, ..., n_samples - 1`` are
            assumed.

        **params : dict of str -> object
            Parameters to pass through to the underlying VAR ``score()`` method.

        Returns
        -------
        score: float
            Mean VAR prediction score (by default R2, see
            ``estimator.scoring_function``).
        """
        check_is_fitted(self)
        X, sample_ids = _check_X_sample_ids(X, sample_ids)
        validate_data(self, X[0], reset=False)

        params_values = list(params.values())
        scores, lengths = [], []
        for sample_id, X_i, *params_values_i in _optional_zip(
            sample_ids, X, *params_values
        ):
            params_i = {k: v for k, v in zip(params, params_values_i) if v is not None}
            score = self.estimators_[sample_id].score(X=X_i, **params_i)
            scores.append(score)
            lengths.append(len(X_i))

        scores = np.array(scores)
        lengths = np.array(lengths)
        score = np.sum(scores * lengths) / np.sum(lengths)
        return score

    def transform(
        self,
        X: np.ndarray | pd.Series,
        sample_ids: np.ndarray | None = None,
    ) -> np.ndarray:
        """Return the learned VAR coefficient features for each sample time series.

        Parameters
        ----------
        X : array-like of shape (n_samples, sequence_length, n_features)
            Array of multiple multivariate time series. If the sequences are different
            length, ``X`` should be a 1D ndarray of dtype object, where each element is
            a 2D array. ``X`` may also be a pandas ``Series`` object, with its index
            encoding each sample ID (e.g. subject ID). All sample time series should
            already be fit with ``fit()`` or ``incremental_fit()``.

        sample_ids : array of shape (n_samples,), default=None
            Array of sample IDs (e.g. subject ID) for each training time series. If
            ``None``, the sample IDs will be extracted from the index of ``X`` in case
            ``X`` is a ``Series``. Otherwise, sample IDs ``0, ..., n_samples - 1`` are
            assumed. All sample IDs should have already been seen during ``fit()`` or
            ``incremental_fit()``.

        Returns
        -------
        coef : array of shape (n_samples, order, n_features, n_features)
            Array of VAR coefficients for each sample time series, to use as features.
        """
        check_is_fitted(self)
        X, sample_ids = _check_X_sample_ids(X, sample_ids)
        validate_data(self, X[0], reset=False)

        for sample_id in sample_ids:
            if sample_id not in self.estimators_:
                raise ValueError(
                    f"Sample ID {sample_id} not seen during fit; "
                    "you probably need to call incremental_fit."
                )

        coef = np.stack([self.estimators_[sample_id].coef_ for sample_id in sample_ids])
        return coef


def _check_X_sample_ids(
    X: np.ndarray | pd.Series,
    sample_ids: np.ndarray | None,
    *,
    unique: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Check input X and sample IDs.

    If X is a pandas Series and sample_ids is None the series index is used as the
    sample ID.

    ``unique = True`` enforces that all sample IDs are unique.
    """
    if _is_series_like(X):
        if sample_ids is None:
            sample_ids = np.asanyarray(X.index)
        X = X.values

    if sp.issparse(X):
        raise TypeError("Sparse X not supported.")

    X = np.asanyarray(X)
    if X.ndim not in {1, 3}:
        raise ValueError(
            f"Invalid X shape {X.shape}; expected 1D array of arrays "
            "or 3D array of shape (n_samples, sequence_length, n_features)."
        )

    if sample_ids is None:
        sample_ids = np.arange(len(X))
    else:
        sample_ids = np.asanyarray(sample_ids)

    if sample_ids.ndim != 1:
        raise ValueError("Expected 1D sample IDs.")
    if unique and len(np.unique(sample_ids)) < len(sample_ids):
        raise ValueError("Sample IDs contain duplicates.")
    if len(sample_ids) != len(X):
        raise ValueError("Lengths of X and sample_ids don't match")

    # Check individual time series in case of a 1D array of arrays.
    if X.ndim == 1:
        X_ = []
        n_features = set()
        for sample_id, X_i in zip(sample_ids, X):
            X_i = check_array(X_i, input_name=f"sample {sample_id}")
            X_.append(X_i)
            n_features.add(X_i.shape[-1])
        if len(n_features) > 1:
            raise ValueError("All samples should have the same number of features")
        X = _stack_arrays(X_)

    return X, sample_ids


def _stack_arrays(arrays: list[np.ndarray]) -> np.ndarray:
    """Stack arrays of possibly different dimensions."""
    try:
        return np.stack(arrays)
    except ValueError:
        # https://stackoverflow.com/a/68824867
        stacked = np.empty(len(arrays), dtype=object)
        stacked[:] = arrays
        return stacked


def _is_series_like(obj: Any) -> bool:
    """Check if an object is a pandas Series or similar."""
    return hasattr(obj, "index") and hasattr(obj, "values")
