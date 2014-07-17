.. _api_ref:

API Reference
=============

Factorization
-------------

.. automodule:: thunder.factorization
   :no-members:
   :no-inherited-members:

.. currentmodule:: thunder

.. autosummary::
	:toctree: generated/
	:template: class.rst

	factorization.PCA
	factorization.ICA
	factorization.SVD


Regression
-------------

.. automodule:: thunder.regression
   :no-members:
   :no-inherited-members:

.. currentmodule:: thunder

.. autosummary::
	:toctree: generated/
	:template: class.rst

	regression.RegressionModel
	regression.TuningModel
	regression.base.MeanRegressionModel
	regression.base.LinearRegressionModel
	regression.base.BilinearRegressionModel
	regression.base.GaussianTuningModel
	regression.base.CircularTuningModel


Clustering
-------------

.. automodule:: thunder.clustering
   :no-members:
   :no-inherited-members:

.. currentmodule:: thunder

.. autosummary::
	:toctree: generated/
	:template: class.rst

	clustering.KMeans


Time Series
-------------

.. automodule:: thunder.timeseries
   :no-members:
   :no-inherited-members:

.. currentmodule:: thunder

.. autosummary::
	:toctree: generated/
	:template: class.rst

	timeseries.Fourier
	timeseries.Stats
	timeseries.Query
	timeseries.CrossCorr
	timeseries.LocalCorr
	

IO
-------------

.. automodule:: thunder.io
   :no-members:
   :no-inherited-members:

.. currentmodule:: thunder

.. autosummary::
	:toctree: generated/
	:template: function.rst

	io.load
	io.save

	:template: class.rst

	io.DataSets
	io.datasets.KMeansData
	io.datasets.PCAData
	io.datasets.FishData
	io.datasets.IrisData


