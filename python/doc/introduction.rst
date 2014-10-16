.. _introduction_ref:

.. currentmodule:: thunder

Introduction to Thunder
=======================

Spatial and temporal data is all around us, whether images from satellites or time series from electronic or biological sensors. These kinds of data are also the bread and butter of neuroscience. Almost all raw neural data consists of electrophysiological time series, or time-varying images of flourescence or resonance.

Thunder is a library for analyzing large spatial and temporal data. Its core components are:

- Methods for loading distributed collections of images and time series data
- Data structures and methods for working with these data types
- Analysis methods for extracting patterns from these data

It is built on top of the `Spark <http://www.spark-project.org/>`_ distributed computing platform. The API is broadly organized into:

- A :class:`ThunderContext` with methods for loading or converting raw input sources
- Classes for distributed data types, like :class:`Images`, :class:`Series`, :class:`TimeSeries`, and :class:`RowMatrix`
- Methods for performing common manipulations, like :func:`Series.normalize` and :func:`Images.subsample`
- Classes for analyses, like :class:`RegressionModel` and :class:`ICA`, with methods for fitting models and extracting results or predictions
- Helper components like :class:`Colorize` and :func:`export` for working with and inspecting analysis results

All data structures are backed by a Resiliant Distributed Dataset, the primary abstraction of Spark, which provides the capability for distributed operation and in-memory cacheing for fast iterative computation or repeated queries. Data attributes, like dimensions, are maintained and lazily propagated to avoid repeated computation.

The codebase is almost entirely Python, and built on Spark's Python API, PySpark. Although Spark itself is written in Scala, PySpark offers enoromous advantages in terms of usability, flexibility, and compatability with external packages, though Python implementations can be slower than pure Scala versions. In some cases, we have implemented routines in Scala (e.g. for file IO) that are called from PySpark; in the future, this could be used to further optimize performance of core analysis components.

The data structures (especially the :class:`Series` class) are heavily inspired by related methods from `Pandas <http://pandas.pydata.org/>`_, but redesigned from scratch for distributed operation. The API for all primary analyses follows the design pattern of `scikit-learn <http://scikit-learn.org/stable/>`_, with methods to ``fit``, ``predict``, and ``transform``. Most of Thunders's methods are distinct from those included with Spark as part of `MLlib`. In cases of overlap, we call `MLlib` implementations under-the-hood, but wrap them in `scikit-learn` style bindings and in some cases add addtional functionality.