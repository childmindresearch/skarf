import logging
import time
from pathlib import Path

import numpy as np
import pytest
from sklearn.utils.estimator_checks import parametrize_with_checks

from arfc import set_cache_dir
from arfc.covariance import pyspi


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

pytestmark = pytest.mark.skipif(
    not pyspi.is_pyspi_available(), reason="PySPI not available"
)


@pytest.fixture(scope="module")
def cache_dir(tmp_path_factory) -> Path:
    path = tmp_path_factory.mktemp("cache")
    set_cache_dir(path)
    return path


@pytest.mark.parametrize("subset", ["all", "fast", "sonnet", "fabfour"])
def test_load_spi_config_map(subset: str, cache_dir: Path):
    # Extract SPI config map and cache
    spi_config_map = pyspi.load_spi_config_map(subset)
    assert isinstance(spi_config_map, dict)

    # Load from cached variable
    assert subset in pyspi._SPI_CONFIG_MAPS
    spi_config_map2 = pyspi.load_spi_config_map(subset)
    assert spi_config_map == spi_config_map2
    del pyspi._SPI_CONFIG_MAPS[subset]

    # Load from cached path
    config_map_path = pyspi._get_spi_config_map_path(subset)
    assert config_map_path.exists()
    spi_config_map3 = pyspi.load_spi_config_map(subset)
    assert spi_config_map == spi_config_map3


@pytest.mark.parametrize("subset", ["all", "fast", "sonnet", "fabfour"])
def test_create_spi_by_name(subset: str, cache_dir: Path):
    available_spis = pyspi.list_available_spis(subset)
    for name in available_spis:
        pyspi.create_spi(name)
        pyspi.create_spi(name=name)


@pytest.mark.parametrize("subset", ["all", "fast", "sonnet", "fabfour"])
def test_create_spi_by_config(subset: str, cache_dir: Path):
    spi_config_map = pyspi.load_spi_config_map(subset)
    for config in spi_config_map.values():
        pyspi.create_spi(config["module_name"], config["fcn"], **config["params"])
        pyspi.create_spi(
            module_name=config["module_name"], fcn=config["fcn"], **config["params"]
        )


def test_spi_covariance(cache_dir: Path):
    rng = np.random.default_rng(42)
    n_samples, n_features = 16, 8
    X = rng.normal(size=(n_samples, n_features))

    for spi in pyspi.list_available_spis(subset="fabfour"):
        cov = pyspi.SPICovariance(spi=spi)
        tic = time.monotonic()
        cov.fit(X)
        rt = time.monotonic() - tic
        assert cov.covariance_.shape == (n_features, n_features)
        nan_count = np.sum(np.isnan(cov.covariance_))
        logger.info("SPI %s: rt=%.3fs, NaNs=%d", spi, rt, nan_count)


@parametrize_with_checks(
    [
        pyspi.SPICovariance("cov_EmpiricalCovariance"),
    ],
    # expected_failed_checks=lambda estimator: {
    #     "check_sample_weight_equivalence_on_dense_data": "binary sample weights only",
    #     "check_sample_weights_list": "binary sample weights only",
    #     "check_sample_weights_not_overwritten": "binary sample weights only",
    # },
)
def test_sklearn_compatible_estimator(estimator, check):
    check(estimator)
