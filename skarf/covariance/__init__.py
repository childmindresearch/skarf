"""Extension covariance estimators following :mod:`sklearn.covariance`."""

from ._pyspi import (
    SPI,
    SPICovariance,
    create_spi,
    create_spi_from_config,
    list_available_spis,
    load_spi_config_map,
    is_pyspi_available,
    load_pyspi_optional_deps,
)
