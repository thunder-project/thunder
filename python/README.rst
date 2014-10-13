.. figure:: https://travis-ci.org/freeman-lab/thunder.png
   :align: left
   :target: https://travis-ci.org/freeman-lab/thunder

thunder
=======

Large-scale neural data analysis with Spark - `project page`_

.. _project page: http://freeman-lab.github.io/thunder/

About
-----

thunder is a library for analyzing large-scale neural data. It's fast to run, easy to develop for, and can be used interactively. It is built on Spark, a new framework for cluster computing.

thunder includes utilties for loading and saving different formats, classes for working with distributed spatial and temporal data, and modular functions for time series analysis, factorization, and model fitting. Analyses can easily be scripted or combined. It is written in Spark's Python API (Pyspark), making use of scipy, numpy, and scikit-learn.

Documentation
-------------
This README contains basic information for installation and usage. See the `documentation`_ for more details, example usage, and API references. If you have a problem, question, or idea, post to the `mailing list`_. If you find a bug, submit an `issue`_. If posting an issue, please provide information about your environment (e.g. local usage or EC2, operating system) and instructions for reproducing the error.

.. _documentation: http://thefreemanlab.com/thunder/docs/
.. _mailing list: https://groups.google.com/forum/?hl=en#!forum/thunder-user
.. _issue: https://github.com/freeman-lab/thunder/issues

Quick start
-----------

thunder is designed to run on a cluster, but local testing is a great way to learn and develop. Many computers can install it with just a few simple steps. If you aren't currently using Python for scientific computing, `Anaconda`_ is highly recommended.

.. _Anaconda: https://store.continuum.io/cshop/anaconda/

1) Download the latest, pre-built version of `Spark`_, and set one environmental variable

.. _Spark: http://spark.apache.org/downloads.html

::

	export SPARK_HOME=/your/path/to/spark

2) Install thunder

:: 

	pip install thunder-python

3) Start thunder from the terminal

:: 

	thunder
	>> from thunder import ICA
	>> data = tsc.makeExample("ica")
	>> model = ICA(c=2).fit(data)

To run in iPython, just set this environmental variable before staring:

::

	export IPYTHON=1

To run analyses as standalone jobs, use the submit script

::

	thunder-submit <package/analysis> <datadirectory> <outputdirectory> <opts>

We also include a script for launching an Amazon EC2 cluster with thunder preinstalled

::

	thunder-ec2 -k mykey -i mykey.pem -s <number-of-nodes> launch <cluster-name>


Analyses
--------

thunder currently includes two primary data types for distributed spatial and temporal data, and four main analysis packages: classification (decoding), clustering, factorization, rand egression, and timeseries. It also provides an entry point for loading and converting a variety of raw data formats, and utilities for exporting or inspecting results. Scripts can be used to run standalone analyses, but the underlying classes and functions can be used from within the PySpark shell for easy interactive analysis.

Input and output
----------------

The primary data types in thunder -- Images and Series -- can each be loaded from a variety of raw input formats, including text or flat binary files (for Series) and tif or pngs (for Images). Files can be stored locally, on a networked file system, on Amazon's S3, or in HDFS. All metadata (e.g. model parameters) can be provided as numpy arrays or loaded from MAT files, see relavant functions for more details.

Results can be visualized directly from the python shell or in iPython notebook, or saved to external formats.

Road map
----------------
If you have other ideas or want to contribute, submit an issue or pull request!
