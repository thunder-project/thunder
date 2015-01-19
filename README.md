Thunder
=======

[![Gitter](https://badges.gitter.im/Join%20Chat.svg)](https://gitter.im/thunder-project/thunder?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge)

<div class="row">
  <a href="http://freeman-lab.github.io/thunder/">
      <img src="http://thefreemanlab.com/thunder/docs/_static/thumbnail_row.png" width="800px" height="125px">
  </a>
</div>

Large-scale neural data analysis with Spark - [project page](http://freeman-lab.github.io/thunder/)

[![Build Status](https://travis-ci.org/freeman-lab/thunder.png?branch=master)](https://travis-ci.org/freeman-lab/thunder) 

About
-----

Thunder is a library for analyzing large-scale spatial and temporal neural data. It's fast to run, easy to extend, and designed for interactivity. It is built on Spark, a new framework for cluster computing.

Thunder includes utilities for loading and saving different formats, classes for working with distributed spatial and temporal data, and modular functions for time series analysis, factorization, and model fitting. Analyses can easily be scripted or combined. It is written against Spark's Python API (Pyspark), making use of scipy, numpy, and scikit-learn.

Documentation
-------------

This README contains info on installation and usage and how to get help. See the complete [documentation](http://thefreemanlab.com/thunder/docs/) for more details, tutorials, and API references. 

Quick start
-----------

Thunder is designed to run on a cluster, but local testing is a great way to learn and develop. Many computers can install it with just a few simple steps. If you aren't currently using Python for scientific computing, [Anaconda](https://store.continuum.io/cshop/anaconda/) is highly recommended.

1) Download the latest "pre-built for Hadoop 1.x" version of [Spark](http://spark.apache.org/downloads.html) and set an environmental variable

	export SPARK_HOME=/your/path/to/spark

2) Install Thunder

	pip install thunder-python

3) Start Thunder from the terminal

	thunder
	>> from thunder import ICA
	>> data = tsc.makeExample("ica")
	>> model = ICA(c=2).fit(data)

To run in iPython, just set this environmental variable before staring:

	export IPYTHON=1

To run analyses as standalone jobs, use the submit script

	thunder-submit <analysis name or script file> <datadirectory> <outputdirectory> <opts>

We also include a script for launching an Amazon EC2 cluster with Thunder preinstalled

	thunder-ec2 -k mykey -i mykey.pem -s <number-of-nodes> launch <cluster-name>


Analyses
--------

Thunder currently includes two primary data types for distributed spatial and temporal data, and four main analysis packages: classification (decoding), clustering, factorization, and regression. It also provides an entry point for loading and converting a variety of raw data formats, and utilities for exporting or inspecting results. Scripts can be used to run standalone analyses, but the underlying classes and functions can be used from within the PySpark shell or an iPython notebook for easy interactive analysis.

Input and output
----------------

The primary data types in Thunder — Images and Series — can each be loaded from a variety of raw input formats, including text or flat binary files (for Series) and tif or pngs (for Images). Files can be stored locally, on a networked file system, on Amazon's S3, or in HDFS. Where needed, metadata (e.g. model parameters) can be provided as numpy arrays or loaded from MAT files. Results can be visualized directly from the python shell or in iPython notebook, or saved to external formats.

Help
------------
We maintain a [chatroom](http://) on gitter. You can also post questions or ideas to the [mailing list](https://groups.google.com/forum/?hl=en#!forum/thunder-user). If you find a reproducible bug, submit an [issue](https://github.com/freeman-lab/thunder/issues). If posting an issue, please provide information about your environment (e.g. local usage or EC2, operating system) and instructions for reproducing the error.


Contributions
-------------
If you have ideas or want to contribute, submit an issue or pull request, or reach out to us on gitter, twitter (@thefreemanlab), or the [mailing list](https://groups.google.com/forum/?hl=en#!forum/thunder-user).
