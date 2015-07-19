.. _api:

API
===

Here we provide an overview of the Thunder's core classes and methods. For example usage, see the :ref:`tutorials`.

Context
-------

.. currentmodule:: thunder.utils.context

.. autosummary::
	:toctree: generated/
	:template: class.rst
	:nosignatures:

	ThunderContext


Data types
----------

.. currentmodule:: thunder.rdds.series

.. autosummary::
	:toctree: generated/
	:template: class.rst
	:nosignatures:

	Series

.. currentmodule:: thunder.rdds.images

.. autosummary::
	:toctree: generated/
	:template: class.rst
	:nosignatures:

	Images


Preprocessing
-------------

.. currentmodule:: thunder.extraction.extraction

.. autosummary::
	:toctree: generated/
	:template: class.rst
	:nosignatures:

	SourceExtraction

.. currentmodule:: thunder.extraction.source

.. autosummary::
	:toctree: generated/
	:template: class.rst
	:nosignatures:

	Source
	SourceModel

.. currentmodule:: thunder.registration.registration

.. autosummary::
	:toctree: generated/
	:template: class.rst
	:nosignatures:

	Registration
	RegistrationModel

.. currentmodule:: thunder.registration.methods.crosscorr

.. autosummary::
	:toctree: generated/
	:template: class.rst
	:nosignatures:

	CrossCorr
	PlanarCrossCorr	

Factorization
-------------

.. currentmodule:: thunder.factorization.nmf

.. autosummary::
	:toctree: generated/
	:template: class.rst
	:nosignatures:

	NMF

.. currentmodule:: thunder.factorization.ica

.. autosummary::
	:toctree: generated/
	:template: class.rst
	:nosignatures:

	ICA

.. currentmodule:: thunder.factorization.pca

.. autosummary::
	:toctree: generated/
	:template: class.rst
	:nosignatures:

	PCA

.. currentmodule:: thunder.factorization.svd

.. autosummary::
	:toctree: generated/
	:template: class.rst
	:nosignatures:

	SVD

Clustering
----------

.. currentmodule:: thunder.clustering.kmeans

.. autosummary::
	:toctree: generated/
	:template: class.rst
	:nosignatures:

	KMeans
	KMeansModel