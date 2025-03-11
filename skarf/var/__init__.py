"""
The :mod:`skarf.var` provides estimators for fitting vector autoregressive (VAR) time
series models.
"""

from ._base import BaseVAR
from ._covariance import CovarianceVAR
from ._linear import LinearVAR
from ._multi import MultiVAR
