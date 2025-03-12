"""A scikit-learn compatible covariance estimator interface for PySPI SPIs."""

import importlib
import logging
import yaml
import traceback
from importlib import resources
from typing import Any, Literal, Protocol, Self, TypeAlias

import numpy as np
from sklearn.base import BaseEstimator, _fit_context
from sklearn.utils.validation import validate_data
from sklearn.utils._param_validation import HasMethods

try:
    import pyspi  # noqa
    from pyspi.data import Data

    _PYSPI_AVAILABLE = True
except ImportError:
    Data: TypeAlias = Any
    _PYSPI_AVAILABLE = False

_logger = logging.getLogger(__name__)

# Mappings of SPI identifiers to configurations, one per subset.
# IMO, it would be nice if PySPI provided a way to instantiate individual SPIs. But it
# seems they do not (see also https://github.com/DynamicsAndNeuralSystems/pyspi/issues/72).
# So as a workaround we provide this functionality.
_SPI_CONFIG_MAP_CACHE = {}


class SPI(Protocol):
    """Abstract minimal interface for PySPI SPI object."""

    def multivariate(self, data: Data) -> np.ndarray:
        """Compute the matrix of pairwise interaction statistics.

        Parameters
        ----------
        data : Data object
            PySPI Data object representing a multivariate time series of shape
            (n_features, n_samples).

        Returns
        -------
        covariance : ndarray
            Matrix of pairwise interaction statistics (i.e. generalized covariance) of
            shape (n_features, n_features).
        """


class SPICovariance(BaseEstimator):
    """Covariance estimator wrapper around a PySPI SPI estimator.

    Parameters
    ----------
    spi : str or SPI object
        Name of SPI or SPI object with `multivariate` method.

    Attributes
    ----------
    covariance_ : ndarray of shape (n_features, n_features)
        Estimated covariance matrix

    n_features_in_ : int
        Number of features seen during `fit`.

    feature_names_in_ : array of shape (`n_features_in_`,)
        Names of features seen during `fit`. Defined only when `X` has feature names
        that are all strings.

    Notes
    -----
    Some of the SPI estimators in PySPI are themselves wrappers around sklearn
    covariance estimators. In those cases, this double wrapping is redundant. We include
    this wrapper however to have a familiar uniform API for all SPIs.
    """

    _parameter_constraints: dict = {
        "spi": [str, HasMethods("multivariate")],
    }

    covariance_: np.ndarray
    """Estimated "covariance" matrix, shape (n_features, n_features)"""

    n_features_in_: int
    """Number of features seen during `fit`."""

    feature_names_in_: np.ndarray
    """Names of features seen during `fit`. Defined only when `X` has feature names."""

    def __init__(self, spi: str | SPI):
        self.spi = spi

    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, X: np.ndarray, y: None = None) -> Self:
        """Fit the underlying PySPI SPI estimator

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
          Training data, where `n_samples` is the number of samples and
          `n_features` is the number of features.

        y : Ignored
            Not used, present for API consistency by convention.

        Returns
        -------
        self : object
            Returns the instance itself.
        """
        _check_is_pyspi_available()
        spi = create_spi(self.spi) if isinstance(self.spi, str) else self.spi
        X = validate_data(self, X, ensure_min_features=2, ensure_min_samples=2)

        data = Data(X.T, normalise=False)
        covariance = spi.multivariate(data)
        self.covariance_ = covariance
        return self


def load_spi_config_map(
    subset: Literal["all", "fast", "sonnet", "fabfour"] = "all",
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    """Extract a mapping of SPI identifiers to configuration dicts.

    Parameters
    ----------
    subset : {'all', 'fast', 'sonnet', 'fabfour'}, default='full'
        PySPI config yaml subset.

    Returns
    -------
    spi_config_map : dict of str -> dict of str -> value
        Dictionary mapping SPI identifier names to SPI config dictionaries. Each config
        dictionary should contain the following keys:

        * 'module_name' : Name of the module where the SPI is defined, relative to\
            `pyspi`. E.g. `'.statistics.basic'`.

        * 'fcn' : PySPI function name (i.e. class name), e.g. `'Covariance'`.

        * 'params' : Parameters passed through to the PySPI SPI `fcn`, e.g.\
            `{'estimator': 'EmpiricalCovariance', 'squared': True}`.

    unavailable_spi_configs : list of dict of str -> value
        List of SPI config dictionaries for unavailable SPIs. SPIs may be unavailable
        for example because of a missing optional python or system dependency.
    """
    # Nb, this config information is not represented statically in the PySPI package
    # anywhere (as far as I can tell, see also
    # https://github.com/DynamicsAndNeuralSystems/pyspi/issues/72). So we need to
    # extract it dynamically, following the code here:
    # https://github.com/DynamicsAndNeuralSystems/pyspi/blob/v1.1.1/pyspi/calculator.py#L212

    # Nb, using this manual cache bc functools cache messes up the function signature.
    # https://github.com/python/typeshed/issues/11280
    if subset in _SPI_CONFIG_MAP_CACHE:
        return _SPI_CONFIG_MAP_CACHE[subset]

    config = _load_pyspi_config_yaml(subset)

    spi_config_map = {}
    unavailable_spi_configs = []

    for module_name in config:
        # Need to import the module bc the SPI identifier is constructed dynamically.
        module = importlib.import_module(module_name, "pyspi")

        for fcn in config[module_name]:
            # If no configs, then it is just the empty config.
            configs = config[module_name][fcn].get("configs") or [{}]
            for params in configs:
                try:
                    # Construct the SPI to get its identifier
                    spi = getattr(module, fcn)(**params)
                    _logger.debug(
                        f"Loaded SPI {spi.identifier}: "
                        f"{module_name=}, {fcn=}, {params=}"
                    )
                    spi_config_map[spi.identifier] = {
                        "module_name": module_name,
                        "fcn": fcn,
                        "params": params,
                    }
                except Exception:
                    _logger.warning(
                        f"Encountered error when loading SPI: "
                        f"{module_name=}, {fcn=}, {params=}\n\n"
                        + traceback.format_exc(limit=0)
                    )
                    unavailable_spi_configs.append(
                        {"module_name": module_name, "fcn": fcn, "params": params}
                    )

    _SPI_CONFIG_MAP_CACHE[subset] = spi_config_map, unavailable_spi_configs

    return spi_config_map, unavailable_spi_configs


def _load_pyspi_config_yaml(
    subset: Literal["all", "fast", "sonnet", "fabfour"] = "all",
):
    """Load PySPI config YAML."""
    name = "config" if subset == "all" else f"{subset}_config"
    with resources.files("pyspi").joinpath(f"{name}.yaml").open() as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config


def list_available_spis(
    subset: Literal["all", "fast", "sonnet", "fabfour"] = "all",
) -> list[str]:
    """List available PySPI SPIs.

    Parameters
    ----------
    subset : {'all', 'fast', 'sonnet', 'fabfour'}, default='full'
        PySPI config yaml subset.

    Returns
    -------
    spi_names : list of str
        List of name identifiers for available SPIs.
    """
    spi_config_map, _ = load_spi_config_map(subset=subset)
    spi_names = list(spi_config_map)
    return spi_names


def create_spi(name: str) -> SPI:
    """Create an SPI by name.

    Parameters
    ----------
    spi : str
        SPI identifier, e.g. `'cov-sq_EmpiricalCovariance'`.

    Returns
    -------
    spifun : `SPI` object.
        Initialized :class:`SPI` object.

    See Also
    --------
    list_available_spis : List all identifiers for available SPIs.
    """
    spi_config_map, _ = load_spi_config_map()
    config = spi_config_map[name]
    module_name = config["module_name"]
    fcn = config["fcn"]
    params = config.get("params") or {}
    return create_spi_from_config(module_name, fcn, **params)


def create_spi_from_config(module_name: str, fcn: str, **params) -> SPI:
    """Create an SPI by its module name and function (with params).

    Parameters
    ----------
    module_name : str
        Name of the module where the SPI is defined, relative to `pyspi`. E.g.
        `'.statistics.basic'`.

    fcn : str
        PySPI function name (i.e. class name), e.g. `'Covariance'`.

    **params : dict of str -> value
        Parameters passed through to the PySPI SPI `fcn`, e.g. `{'estimator':
        'EmpiricalCovariance', 'squared': True}`.

    Returns
    -------
    spifun : `SPI` object.
        Initialized :class:`SPI` object.

    See Also
    --------
    load_spi_config_map : Load the SPI config map of SPI identifiers to config dicts.
    """
    module = importlib.import_module(module_name, "pyspi")
    spi = getattr(module, fcn)(**params)
    return spi


def is_pyspi_available() -> bool:
    """Check if PySPI is installed."""
    return _PYSPI_AVAILABLE


def _check_is_pyspi_available() -> None:
    if not is_pyspi_available():
        raise ModuleNotFoundError(
            "PySPI required, please install by visiting "
            "https://github.com/DynamicsAndNeuralSystems/pyspi)"
        )
