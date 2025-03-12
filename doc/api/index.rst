.. _api:

#############
API reference
#############

This is the full API documentation of the `skarf` toolbox.

Vector autoregressive (VAR) models
==================================

.. automodule:: skarf.var
    :no-members:
    :no-inherited-members:

Covariance based VAR models
---------------------------

.. automodule:: skarf.var._covariance
   :no-members:
   :no-inherited-members:

.. currentmodule:: skarf.var

.. autosummary::
   :toctree: generated/
   :template: class.rst

   CovarianceVAR


Regularized linear VAR models
-----------------------------

.. automodule:: skarf.var._linear
   :no-members:
   :no-inherited-members:

.. currentmodule:: skarf.var

.. autosummary::
   :toctree: generated/
   :template: class.rst

   LinearVAR


Multi-sample VAR models
-----------------------

.. automodule:: skarf.var._multi
   :no-members:
   :no-inherited-members:

.. currentmodule:: skarf.var

.. autosummary::
   :toctree: generated/
   :template: class.rst

   MultiVAR


Covariance estimators
=====================

.. automodule:: skarf.covariance
    :no-members:
    :no-inherited-members:

PySPI covariance estimators
---------------------------

.. automodule:: skarf.covariance._pyspi
   :no-members:
   :no-inherited-members:

.. currentmodule:: skarf.covariance

.. autosummary::
   :toctree: generated/
   :template: class.rst

   SPI
   SPICovariance

.. autosummary::
   :toctree: generated/
   :template: functions.rst

   create_spi
   create_spi_from_config
   list_available_spis
   load_spi_config_map
