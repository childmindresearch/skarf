# Autoregressive Functional Connectivity (ARFC)

This package provides regularized linear vector autoregressive models of functional connectivity, implemented as scikit-learn estimators. We provide two kinds of models:

- [covariance AR models](arfc/covariance.py): these take a fixed [covariance](https://scikit-learn.org/stable/modules/covariance.html) matrix (or more generally any fixed similarity matrix) and convert it to a linear AR model by way of an optimized polynomial transform (with optional ridge regularization).
- [regularized linear AR models](arfc/linear_model.py): these are standard [regularized linear regression models](https://scikit-learn.org/stable/modules/linear_model.html) applied to the task of multi-lag vector time series autoregression.

In addition, we provide a [multi-subject meta AR model](arfc/multi.py) which fits an independent AR model for each subject in a dataset.
