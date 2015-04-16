.. _contributing:

.. currentmodule:: thunder

Contributing
============

With Thunder we want to provide a centralized library to develop and vet analyses. If you want to contribute, follow these steps to download and use the source code. If you have a contribution, submit a `pull request <https://github.com/freeman-lab/thunder/pulls>`_. If you find a problem, submit an `issue <https://github.com/freeman-lab/thunder/issues>`_!

Setting up
~~~~~~~~~~
For development, you'll want to clone the repository from github (or fork your own branch):

.. code-block:: bash

	git clone git://github.com/freeman-lab/thunder.git

Then set two additional environmental variables to make sure the code and executables are on your paths (we assume you cloned into a directory called ``code``)

.. code-block:: bash

	export PYTHONPATH=~/code/thunder/python/:$PYTHONPATH
	export PATH=~/code/thunder/python/bin:$PATH

To avoid confusion, if you had already installed Thunder previously using ``pip``, we recommend uninstalling it first using ``pip uninstall thunder-python``. 

Finally, install Thunder's dependencies manually by callling `pip install -r ~/code/thunder/python/requirements.txt` (you may have many of these already).

Using an IDE for development is highly recommended, we use `PyCharm <http://www.jetbrains.com/pycharm/>`_ for Python and `IntelliJ <http://www.jetbrains.com/idea/>`_ for Scala. And here's a good `article <https://gun.io/blog/how-to-github-fork-branch-and-pull-request/>`_ on how to contribute a pull request to a project using github.

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

You should write a test for anything new that you develop, as well as make sure the existing tests still pass (automated testing will be performed on any pull request using Travis). If you're unsure how to write a new test, see the existing ones for examples.

Packaging a release
~~~~~~~~~~~~~~~~~~~
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
A good starting point is to check the `issue <https://github.com/freeman-lab/thunder/issues>`_ page. If there is an outstanding issue that appears unaddressed, add a comment that you are starting to work on it.

Beyond existing issues, the following are fairly simple new features that would make for a great initial contribution:

- Add new temopral filtering methods to :class:`TimeSeries`, like moving average or savistky golay
- Add more time series calculations to :class:`TimeSeries`, like autocorrelation
- Implement shuffle / permutation tests for :class:`RegressionModel` analyses
- Add more basic image processing routines to :class:`Images`, like blurring or blob detection

These are more involved. It would be worth posting to the `gitter chatroom <https://gitter.im/thunder-project/thunder>`_ or `mailing list <https://groups.google.com/forum/?hl=en#!forum/thunder-user>`_ before starting work on them to avoid duplcating other efforts.

- A new EventSeries class and associated methods for working with event data
- Add a linear discriminant analysis (including LDA for dimensionality reduction)
- Methods for computing and working with arbitrary distance matrices


