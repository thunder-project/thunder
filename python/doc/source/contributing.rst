.. _contributing:

.. currentmodule:: thunder

Contributing
============

Thunder is a community effort and we actively encourage new features and algorithms. Many of the components are designed to make it easy to plug in new capabilities, especially the ``Registration`` and ``SourceExtraction`` APIs.

If you have a contribution, submit a `pull request <https://github.com/freeman-lab/thunder/pulls>`_. If you find a problem, submit an `issue <https://github.com/freeman-lab/thunder/issues>`_!

This section contains information on setting up a local development environment, running unit tests, and the style guidelines we follow.

Setting up
~~~~~~~~~~
For development, you'll want to fork your own branch of the GitHub repository and then clone it locally:

.. code-block:: bash

	git clone git://github.com/your-user-name/thunder.git

Then set two additional environmental variables to make sure the code and executables are on your paths (we assume you cloned into a directory called ``~/code``)

.. code-block:: bash

	export PYTHONPATH=~/code/thunder/python/:$PYTHONPATH
	export PATH=~/code/thunder/python/bin:$PATH

Finally, manually install Thunder's dependencies (if neccessary) by callling 

.. code-block:: bash

	pip install -r ~/code/thunder/python/requirements.txt

To avoid confusion, if you had previously installed the release version of Thunder using ``pip install thunder-python``, we recommend uninstalling it first using ``pip uninstall thunder-python`` before performing the steps above.

Using an IDE for development is highly recommended, many of us use `PyCharm <http://www.jetbrains.com/pycharm/>`_ for Python and `IntelliJ <http://www.jetbrains.com/idea/>`_ for Scala. If you are new to contributing to open source software, here's an `article <https://gun.io/blog/how-to-github-fork-branch-and-pull-request/>`_ on how to create a pull request, and also check out the `gitgoing <https://github.com/CodeNeuro/gitgoing>`_ tutorial.

Running the tests 
~~~~~~~~~~~~~~~~~
You can run the unit tests with ``nose`` by calling

.. code-block:: bash
	
	cd ~/code/thunder/python/test
	./run_tests.sh

You must run the tests from within this folder. You can run just one set of tests, or a single test within a suite, like this

.. code-block:: bash

	./run_tests.sh test_timeseries.py
	./run_tests.sh test_timeseries.py:TestStats

Within the ``thunder/test`` folder, there are several files with names beginning with ``test_`` each containing unit tests for a corresponding package or module. Within each of these files, there are one or more test classes (usually derived from the base class ``PySparkTestCase``), and each test class has as methods a set of individual tests.

New features should include appropriate unit tests. When adding a new feature, a good place to start is to find a piece of functionality similar to the one you are adding, find the existing test, and use it as as a starting point.

All tests will be automatically run on any pull request, but you can save time by running tests locally and resolving any issues before submitting PRs.

Conventions
~~~~~~~~~~~

Naming
^^^^^^^^^^

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
^^^^^^^

All imports should be in the form ``from modulename import a, b, c``. This includes ``numpy``. Please do not use the convention of ``import numpy as np``. Although there are some advantages to this approach, we find overall that it's harder to read.

Imports should be grouped as follows: all third-party imports (e.g. ``numpy``, ``scipy``), then all ``thunder`` imports, with alphabetical sorting by module within each group. And variables imported from a module should be listed in alphabetical order. For example

.. code-block:: python

	from numpy import arange, amax, amin, dtype, greater, ndarray, size, squeeze

	from thunder.rdds.data import Data
	from thunder.rdds.keys import Dimensions
	from thunder.utils.common import parseMemoryString

Locally-scoped imports are strongly encouraged, and top-level imports discouraged, especially for modules with user-facing classes (anything imported in Thunder's top-level ``__init__.py``, which are those modules imported when calling ``from thunder import *``). This is to avoid a performance problem arising from how PySpark serializes inline functions, especially problematic for imports from large and complex libraries (e.g. ``matplotlib``). For user-facing modules, limit top-level imports to ``numpy``, and other ``thunder`` modules which themselves only import from ``numpy``, and otherwise use locally-scoped imports.

Docstrings
^^^^^^^^^^

Thunder uses the ``scipy`` and ``numpy`` docstring formatting conventions. All user-facing classes and methods should have a description of what the method or class does, a list of any parameters, and optionally a list of returns. Single line docstrings should use this formatting:

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

Design principles
^^^^^^^^^^^^^^^^^

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


Release packaging
^^^^^^^^^^^^^^^^^

(NOTE: this is primarily a reference for core Thunder committers). When we want to create a new major Thunder release, first create a branch with the major version number (e.g. "branch-0.6"). Change the version number in ``python/thunder/__init__.py`` by setting:

.. code-block:: python

	__version__ = "0.6.0"

Also change the version in the `scala/build.sbt` file:

.. code-block:: scala

	version := "0.6.0"

Build the Scala jar by calling from within the ``scala`` folder:

.. code-block:: bash

	sbt package

and copy the jar it creates into ``python/thunder/lib``. Build and copy the Python egg file by calling:

.. code-block:: bash

	python/bin/build

Finally, from within the Python folder, submit to PyPi using:

.. code-block:: bash

	./setup.py sdist upload

Ideas for contributions
~~~~~~~~~~~~~~~~~~~~~~~
A good starting point is to check the `issue <https://github.com/freeman-lab/thunder/issues>`_ page, and if you are just getting started, look for issues with the ``beginner-friendly`` tag. If there is an outstanding issue that appears unaddressed, add a comment that you are starting to work on it. 

For larger additions, come tell us your idea in our `chatroom <https://gitter.im/thunder-project/thunder>`_ and we'll discuss it further!


