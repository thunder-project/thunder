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

thunder includes utilties for data loading and saving, and modular functions for time series statistics, matrix decompositions, and fitting algorithms. Analyses can easily be scripted or combined. It is written in Spark's Python API (Pyspark), making use of scipy, numpy, and scikit-learn.

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
	>> from thunder.factorization import ICA
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

thunder currently includes five packages: classification (decoding), clustering, factorization, regression, and timeseries, as well as an utils package for loading and saving (see Input format and Output format) and other utilities (e.g. matrix operations). Scripts can be used to run standalone analyses, but the underlying classes and functions can be used from within the PySpark shell for easy interactive analysis.

Input and output
----------------

thunder is built around a common input format for time series data: a set of signals or channels as key-value pairs, where the key is an identifier, and the value is a time series. In neural imaging data, for example, each record would be a voxel or an ROI, the key an xyz coordinate, and the value a flouresence time series.

These key-value records can be derived from a variety of cluster-accessible formats. thunder currently includes methods for loading data from text or flat binary files stored locally, in HDFS, or on a networked file system, and preliminary support for importing and converting data from other formats.

All metadata (e.g. parameters of the stimulus or behavior for regression analyses) can be provided as numpy arrays or loaded from MAT files, see relavant functions for more details.

Results can be visualized directly from the python shell or in iPython notebook, or saved as images or MAT files. Other output formats coming soon. 

Road map
----------------
If you have other ideas or want to contribute, submit an issue or pull request!

- Integrate more scikit learn functionality
- Analysis-specific visualizations
- Input format support: HDF5, tif
- Port versions of most common workflows to scala
- Unified metadata representation
- Streaming analyses
