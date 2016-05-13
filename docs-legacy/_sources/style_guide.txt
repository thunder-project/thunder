.. _contributing:

.. currentmodule:: thunder

Style guide
===========

Developers and contributors to Thunder should keep these style and programming conventions in mind.

Naming conventions
~~~~~~~~~~~~~~~~~~

Class names should be ``CapitalizedNames``.

Variables, attributes, methods, and function names should all be ``camelCased``. This is primarily to maintain consistency with the Spark code base. The following are possible exceptions to the convention: 

- variables storing counts beginning with ``n`` (e.g. ``nfeatures``) should be in lower case
- compound words or words sometimes written as one can be in lowercase (e.g. ``filename``, or ``pinv`` for pseudoinverse)

Literal constants should be named in all caps, with underscores separating words (e.g. ``INIT_IFD_SIZE``).

If a method directly wraps a method from another library, under most circumstances the name of the external library should be used regardless of these conventions (e.g. the ``Data.astype`` method, which calls ``numpy.astype``).

Variable names may use abbreviations so long as their meaning is relatively clear, e.g. ``whtMat`` or ``whiteningMatrix`` would be acceptable. Single letter variable names are acceptable if following a known convention, e.g. ``w`` for weights, ``i``, ``j``, ``k`` for loop indices, ``n`` for counts, ``x``, ``y`` for numeric values.

Leading underscores may be used to indicate attributes not intended to be accessed outside of the object where they are defined. Double leading underscores may be used in method names for the same purpose. Trailing underscores may be used (sparingly) to avoid shadowing variables of the same name inside nested scopes, such as nested functions. A single underscore ``_`` in a function may be used to denote a dummy variable required for unpacking a tuple, as in a method return value, that will not be otherwise accessed, e.g. ``data.apply(lambda (_, v): (_, v * 2)``).

A Spark broadcast variable may be denoted by initial lowercase letters ``bc``, as in ``bcModel = sc.broadcast(model)``.

Imports
~~~~~~~

All imports should be in the form ``from modulename import a, b, c``. This includes ``numpy``. Please do not use the convention of ``import numpy as np``. Although there are some advantages to this approach, we find overall that it's harder to read.

Imports should be grouped as follows: all third-party imports (e.g. ``numpy``, ``scipy``), then all ``thunder`` imports, with alphabetical sorting by module within each group. And variables imported from a module should be listed in alphabetical order. For example

.. code-block:: python

	from numpy import arange, amax, amin, dtype, greater, ndarray, size, squeeze

	from thunder.rdds.data import Data
	from thunder.rdds.keys import Dimensions
	from thunder.utils.common import parseMemoryString

Locally-scoped imports are strongly encouraged, and top-level imports discouraged, especially for modules with user-facing classes (anything imported in Thunder's top-level ``__init__.py``, which are those modules imported when calling ``from thunder import *``). This is to avoid a performance problem arising from how PySpark serializes inline functions, especially problematic for imports from large and complex libraries (e.g. ``matplotlib``). For user-facing modules, limit top-level imports to ``numpy``, and other ``thunder`` modules which themselves only import from ``numpy``, and otherwise use locally-scoped imports.

Docstrings
~~~~~~~~~~

Thunder uses the `scipy` and `numpy` docstring formatting conventions. All user-facing classes and methods should have a description of what the method or class does, a list of any parameters, and optionally a list of returns. Single line docstrings should use this formatting:

.. code-block:: python

	""" This is a docstring """

Multi line docstrings should instead use this formatting (note the initial line break):

.. code-block:: python

	""" 
	This is the first line of the docstring.

	This is further information, usually a more in-depth discussion
	of what the method does.
	"""

Use the following conventions to specify parameters, including type and default value:

.. code-block:: python

	"""
	Parameters
	----------
	parameter1 : type
		Description of parameter

	parameter2 : type, optional, default = something
		Description of an optional parameters
	"""

Testing
~~~~~~~

Within the ``thunder/test`` folder, there are several files with names beginning with ``test_`` each containing unit tests for a corresponding package or module. Within each of these files, there are one or more test classes (usually derived from the base class ``PySparkTestCase``), and each test class has as methods a set of individual tests.

All new features should include appropriate unit tests. When adding a new feature, a good place to start is to find a piece of functionality similar to the one you are adding, find the existing test, and use it as as a starting point.

See :doc:`contributing` for information on how to run the tests. All tests will be automatically run on any pull request, but you can save time by running tests locally and resolving any issues before submitting.

Design principles
~~~~~~~~~~~~~~~~~

Most functionality in Thunder is organized broadly into two parts: distributed data objects and their associated methods (e.g. ``Images``, ``Series``, etc.), and analysis packages (e.g. ``clustering``, ``factorization``, ``regression``) that take these objects as inputs.

Methods on data objects are designed to provide easy access to common data processing operations. For example, filtering or cropping images, or computing correlations on time series. Data object methods are not intended for complex analyses.

Analysis packages are designed for more complex operations and model fitting workflows. Most are designed after the style of scikit-learn, with a single class or classes for each kind of model. These classes should usually have ``fit`` methods (for fitting the parameters of a model from a data object), and ``predict`` and/or ``transform`` methods (for applying the estimated model to new data). 

Data objects are extensible by design -- see, for example, ``TimeSeries`` which is a subclass of ``Series`` -- and some new features may be best implemented through a new class (for example, an ``EventTimeSeries`` for working with sparse time series data). To extend an existing data object, your subclass just needs to define a ``_constructor`` that returns the correct type, as in:

.. code-block:: python

  def _constructor(self):
  	return TimeSeries

It can also optionally add metadata (and augment the ``__init__`` method accordingly), as in:

.. code-block:: python

	metadata = Data._metadata + ['_index', '_dims']

.. code-block:: python

  def __init__(self, rdd, index=None, dims=None, dtype=None):
    super(Series, self).__init__(rdd, dtype=dtype)
    self._index = index
    self._dims = dims

All data objects in Thunder (e.g. ``Images``, ``Series``, etc.) are backed by RDDs. These objects do not expose all RDD operations, to provide users with a coherent and intuitive set of custom data manipulations. But as a developer, a common pattern is to call a sequence of arbitrary RDD operations on the object’s underlying RDD, and then reconstitute the class. In the case of a ``map`` operation, this can and should be performed using the ``apply``, ``applyValues``, or ``applyKeys`` methods, which combine that sequence into a single method. For other operations, you will need to perform that sequence as follows:

.. code-block:: python

	newrdd = self.rdd.func1().func2() ...
	newdata = self._constructor(newrdd).__finalize__(self)
	return newdata

Keep these design considereations in mind when planning to add a new feature. It's also a great idea to post an `issue <https://github.com/freeman-lab/thunder/issues>`_, or send a message to the `gitter chatroom <https://gitter.im/thunder-project/thunder>`_ or `mailing list <https://groups.google.com/forum/?hl=en#!forum/thunder-user>`_ with your idea, to solicit feedback from the community. We can help!


