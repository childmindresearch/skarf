import logging
import time
from pathlib import Path

import numpy as np
import pytest
from sklearn.utils.estimator_checks import parametrize_with_checks

from skarf import set_cache_dir
from skarf.covariance import _pyspi


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

pytestmark = pytest.mark.skipif(
    not _pyspi.is_pyspi_available(), reason="PySPI not available"
)


@pytest.fixture(scope="module")
def cache_dir(tmp_path_factory) -> Path:
    path = tmp_path_factory.mktemp("cache")
    set_cache_dir(path)
    return path


@pytest.fixture(scope="module")
def pyspi_deps() -> dict[str, bool]:
    # Loading deps necessary for importing some of the modules during config map
    # extraction.
    deps = _pyspi.load_pyspi_optional_deps()
    return deps


@pytest.mark.parametrize("subset", ["all", "fast", "sonnet", "fabfour"])
def test_load_spi_config_map(subset: str, cache_dir: Path, pyspi_deps: dict[str, bool]):
    # Extract SPI config map
    spi_config_map, _ = _pyspi.load_spi_config_map(subset)
    assert isinstance(spi_config_map, dict)
    assert "cov_EmpiricalCovariance" in spi_config_map

    # Load from in memory cache
    assert subset in _pyspi._SPI_CONFIG_MAP_CACHE
    spi_config_map2, _ = _pyspi.load_spi_config_map(subset)
    assert spi_config_map == spi_config_map2

    # Load from file cache
    del _pyspi._SPI_CONFIG_MAP_CACHE[subset]
    spi_config_map3, _ = _pyspi.load_spi_config_map(subset)
    assert spi_config_map == spi_config_map3


@pytest.mark.parametrize("subset", ["all", "fast", "sonnet", "fabfour"])
def test_create_spi(subset: str, cache_dir: Path, pyspi_deps: dict[str, bool]):
    available_spis = _pyspi.list_available_spis(subset)
    for name in available_spis:
        _pyspi.create_spi(name)


@pytest.mark.parametrize("subset", ["all", "fast", "sonnet", "fabfour"])
def test_create_spi_from_config(
    subset: str, cache_dir: Path, pyspi_deps: dict[str, bool]
):
    spi_config_map, _ = _pyspi.load_spi_config_map(subset)
    for config in spi_config_map.values():
        _pyspi.create_spi_from_config(
            config["module_name"], config["fcn"], config["params"]
        )


def test_spi_covariance(cache_dir: Path, pyspi_deps: dict[str, bool]):
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
