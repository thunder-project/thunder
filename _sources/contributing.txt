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

Ideas for contributions
~~~~~~~~~~~~~~~~~~~~~~~
The following are fairly simple extensions that would make for a great initial contribution:

- Add new temopral filtering methods to :class:`TimeSeries`, like moving average or savistky golay
- Add more time series calculations to :class:`TimeSeries`, like autocorrelation
- Implement shuffle / permutation tests for :class:`RegressionModel` analyses
- Add more basic image processing routines to :class:`Images`, like blurring or blob detection

These are more involved. It would be worth posting to the `mailing list <https://groups.google.com/forum/?hl=en#!forum/thunder-user>`_ before starting work on them to avoid duplcating other efforts.

- A new EventSeries class and associated methods for working with event data
- A basic image alignment routine
- Add a linear discriminant analysis (including LDA for dimensionality reduction)
- Methods for computing and working with arbitrary distance matrices

You can also check the `issue <https://github.com/freeman-lab/thunder/issues>`_ page for contribution ideas.
