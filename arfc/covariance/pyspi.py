import importlib
import logging
import yaml
import traceback
from importlib import metadata, resources
from pathlib import Path
from typing import Any, Literal, Protocol, Self, overload

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.utils.validation import validate_data

from arfc import get_cache_dir

try:
    import pyspi  # noqa
    from pyspi.data import Data

    _PYSPI_AVAILABLE = True
except ImportError:
    _PYSPI_AVAILABLE = False

_logger = logging.getLogger(__name__)

# Mappings of SPI identifiers to configurations, one per subset.
# IMO, it would be nice if PySPI provided a way to instantiate individual SPIs. But it
# seems they do not (see also https://github.com/DynamicsAndNeuralSystems/pyspi/issues/72).
# So as a workaround we provide this functionality.
_SPI_CONFIG_MAPS = {}


class SPI(Protocol):
    """Abstract minimal interface for PySPI SPI object."""

    def multivariate(self, data: Data) -> np.ndarray:
        ...


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

    Notes
    -----
    Some of the SPI estimators in PySPI are themselves wrappers around sklearn
    covariance estimators. In those cases, this double wrapping is redundant. We include
    this wrapper however to have a familiar uniform API for all SPIs.
    """

    covariance_: np.ndarray

    def __init__(self, spi: str | SPI):
        self.spi = spi

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
        X = validate_data(self, X)
        data = Data(X.T, normalise=False)
        covariance = spi.multivariate(data)
        self.covariance_ = covariance
        return self


def is_pyspi_available() -> bool:
    """Check if PySPI is installed

    https://github.com/DynamicsAndNeuralSystems/pyspi
    """
    return _PYSPI_AVAILABLE


def _check_is_pyspi_available() -> None:
    if not is_pyspi_available():
        raise ModuleNotFoundError(
            "PySPI required, please install by visiting "
            "https://github.com/DynamicsAndNeuralSystems/pyspi)"
        )


def _extract_spi_config_map(
    subset: Literal["all", "fast", "sonnet", "fabfour"] = "all",
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Extract a mapping of SPI identifiers to configuration.

    The configuration for each SPI includes the module name, the SPI function, and the
    keyword parameters.

    Returns the SPI config mapping and a list of any configs that failed to load.
    """
    # Nb, this config information is not represented statically in the PySPI package
    # anywhere (as far as I can tell, see also
    # https://github.com/DynamicsAndNeuralSystems/pyspi/issues/72). So we need to
    # extract it dynamically, following the code here:
    # https://github.com/DynamicsAndNeuralSystems/pyspi/blob/v1.1.1/pyspi/calculator.py#L212

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

    return spi_config_map, unavailable_spi_configs


def _get_pyspi_version():
    """Get the installed version of PySPI."""
    _check_is_pyspi_available()
    return metadata.version("pyspi")


def _load_pyspi_config_yaml(
    subset: Literal["all", "fast", "sonnet", "fabfour"] = "all",
):
    """Load PySPI config YAML."""
    name = "config" if subset == "all" else f"{subset}_config"
    with resources.files("pyspi").joinpath(f"{name}.yaml").open() as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config


def _get_spi_config_map_path(
    subset: Literal["all", "fast", "sonnet", "fabfour"] = "all",
) -> Path:
    return get_cache_dir() / f"pyspi_spi_config_map_{subset}.yaml"


def load_spi_config_map(
    subset: Literal["all", "fast", "sonnet", "fabfour"] = "all",
    cache: bool | None = True,
) -> dict[str, Any]:
    """Load PySPI SPI config map, possibly from a cached YAML file."""
    if cache and subset in _SPI_CONFIG_MAPS:
        return _SPI_CONFIG_MAPS[subset]

    pyspi_version = _get_pyspi_version()
    path = _get_spi_config_map_path(subset)

    if cache and path.exists():
        with path.open() as f:
            spi_config_map_yaml = yaml.safe_load(f)

        if spi_config_map_yaml["__pyspi_version__"] != pyspi_version:
            _logger.info(
                "PySPI SPI config map doesn't match installed PySPI version "
                f"{pyspi_version}; removing."
            )
            path.unlink()
        else:
            _logger.info("Loaded PySPI SPI config map from cache: %s", path)
            spi_config_map = spi_config_map_yaml["configs"]
            _SPI_CONFIG_MAPS[subset] = spi_config_map
            return spi_config_map

    spi_config_map, _ = _extract_spi_config_map(subset=subset)
    _SPI_CONFIG_MAPS[subset] = spi_config_map

    if cache or cache is None:
        _logger.info("Caching PySPI SPI config map to %s", path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w") as f:
            spi_config_map_yaml = {
                "__pyspi_version__": pyspi_version,
                "configs": spi_config_map,
            }
            yaml.safe_dump(spi_config_map_yaml, f)

    return spi_config_map


def list_available_spis(subset: Literal["all", "fast", "sonnet", "fabfour"] = "all"):
    """List available PySPI SPIs."""
    spi_config_map = load_spi_config_map(subset=subset)
    return list(spi_config_map)


@overload
def create_spi(name: str) -> SPI:
    """Create an SPI by name."""
    ...


@overload
def create_spi(module_name: str, fcn: str, **params) -> SPI:
    """Create an SPI by its module name and function (with params)."""
    ...


def create_spi(*args, **kwargs) -> SPI:
    if len(args) == 1 and len(kwargs) == 0:
        return _create_spi_by_name(args[0])
    elif len(args) == 0 and set(kwargs) == {"name"}:
        return _create_spi_by_name(kwargs["name"])
    else:
        return _create_spi_by_config(*args, **kwargs)


def _create_spi_by_name(name: str) -> SPI:
    spi_config_map = load_spi_config_map()
    config = spi_config_map[name]
    module_name = config["module_name"]
    fcn = config["fcn"]
    params = config.get("params") or {}
    return _create_spi_by_config(module_name, fcn, **params)


def _create_spi_by_config(module_name: str, fcn: str, **params) -> SPI:
    module = importlib.import_module(module_name, "pyspi")
    spi = getattr(module, fcn)(**params)
    return spi
