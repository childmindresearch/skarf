import logging
import time

import numpy as np
import pytest
from sklearn.utils.estimator_checks import parametrize_with_checks

from skarf.covariance import _pyspi


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

pytestmark = pytest.mark.skipif(
    not _pyspi.is_pyspi_available(), reason="PySPI not available"
)


@pytest.mark.parametrize("subset", ["all", "fast", "sonnet", "fabfour"])
def test_load_spi_config_map(subset: str):
    # Extract SPI config map
    spi_config_map, _ = _pyspi.load_spi_config_map(subset)
    assert isinstance(spi_config_map, dict)
    assert "cov_EmpiricalCovariance" in spi_config_map

    # Load from cache
    assert subset in _pyspi._SPI_CONFIG_MAP_CACHE
    spi_config_map2, _ = _pyspi.load_spi_config_map(subset)
    assert spi_config_map == spi_config_map2


@pytest.mark.parametrize("subset", ["all", "fast", "sonnet", "fabfour"])
def test_create_spi(subset: str):
    available_spis = _pyspi.list_available_spis(subset)
    for name in available_spis:
        _pyspi.create_spi(name)


@pytest.mark.parametrize("subset", ["all", "fast", "sonnet", "fabfour"])
def test_create_spi_from_config(subset: str):
    spi_config_map, _ = _pyspi.load_spi_config_map(subset)
    for config in spi_config_map.values():
        _pyspi.create_spi_from_config(
            config["module_name"], config["fcn"], **config["params"]
        )


def test_spi_covariance():
    rng = np.random.default_rng(42)
    n_samples, n_features = 16, 8
    X = rng.normal(size=(n_samples, n_features))

    for spi in _pyspi.list_available_spis(subset="fabfour"):
        cov = _pyspi.SPICovariance(spi=spi)
        tic = time.monotonic()
        cov.fit(X)
        rt = time.monotonic() - tic
        assert cov.covariance_.shape == (n_features, n_features)
        nan_count = np.sum(np.isnan(cov.covariance_))
        logger.info("SPI %s: rt=%.3fs, NaNs=%d", spi, rt, nan_count)


@parametrize_with_checks(
    [
        _pyspi.SPICovariance("cov_EmpiricalCovariance"),
    ],
)
def test_sklearn_compatible_estimator(estimator, check):
    check(estimator)
