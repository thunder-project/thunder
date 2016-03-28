
Thunder
=======

<div class="row">
  <a href="http://thunder-project.org">
      <img src="http://thunder-project.org/thunder/docs/_static/thumbnail_row.png" width="800px" height="125px">
  </a>
</div>

Large-scale image and time series analysis with Spark - [project page](http://thunder-project.org)

[![Latest Version](https://img.shields.io/pypi/v/thunder-python.svg)](https://pypi.python.org/pypi/thunder-python)
[![Build Status](https://img.shields.io/travis/thunder-project/thunder/master.svg)](https://travis-ci.org/thunder-project/thunder) 
[![Gitter](https://badges.gitter.im/Join%20Chat.svg)](https://gitter.im/thunder-project/thunder?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge)

About
-----

Thunder is a library for analyzing large-scale spatial and temporal data. It's fast to run, easy to extend, and designed for interactivity. It is built on Spark, a new framework for cluster computing.

Thunder includes utilities for loading and saving different formats, classes for working with distributed spatial and temporal data, and modular functions for time series analysis, factorization, and model fitting. Analyses can easily be scripted or combined. It is written against Spark's Python API (Pyspark), making use of scipy, numpy, and scikit-learn.

Documentation
-------------

This README contains basic info on installation and usage and how to get help. See the complete [documentation](http://thunder-project.org/thunder/docs) for more details, tutorials, and API references. We also maintain  separate [development documentation](http://thunder-project.org/thunder/docs-dev) for reference if you are running on Thunder's master branch. 

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

To run in IPython, just set this environmental variable before staring:

	export IPYTHON=1

To run analyses as standalone jobs, use the submit script

	thunder-submit <analysis name or script file> <datadirectory> <outputdirectory> <opts>

We also include a script for launching an Amazon EC2 cluster with Thunder preinstalled

	thunder-ec2 -k mykey -i mykey.pem -s <number-of-nodes> launch <cluster-name>


Analyses
--------

Thunder currently includes two primary data types for distributed spatial and temporal data, and five main analysis packages: classification (decoding), clustering, factorization, image processing, and regression. It also provides an entry point for loading and converting a variety of raw data formats, and utilities for exporting or visually inspecting results. Scripts can be used to run standalone analyses, but the underlying classes and functions can be used from within the PySpark shell or an IPython notebook for easy interactive analysis.

Input and output
----------------

The primary data types in Thunder — Images and Series — can each be loaded from a variety of raw input formats, including text or flat binary files (for Series) and binary, tifs, or pngs (for Images). Files can be stored locally, on a networked file system, on Amazon's S3, on Google Storage, or in HDFS. Where needed, metadata (e.g. model parameters) can be provided as numpy arrays or loaded from JSON or MAT files. Results can be visualized directly from the python shell or in IPython notebook using matplotlib, seaborn, or a new interactive visualization library we are developing called [lightning](http://lightning-viz.org)

Help
------------
We maintain a [chatroom](https://gitter.im/thunder-project/thunder?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge) on gitter. You can also post questions or ideas to the [mailing list](https://groups.google.com/forum/?hl=en#!forum/thunder-user). If you find a reproducible bug, submit an [issue](https://github.com/thunder-project/thunder/issues). If posting an issue, please provide information about your environment (e.g. local usage or EC2, operating system) and instructions for reproducing the error.


Contributions
-------------
Thunder is a community effort, and thus far features contributions from the following individuals:

Andrew Osheroff, Ben Poole, Chris Stock, Davis Bennett, Jascha Swisher, Jason Wittenbach, Jeremy Freeman, Josh Rosen, Kunal Lillaney, Logan Grosenick, Matt Conlen, Michael Broxton, Noah Young, Ognen Duzlevski, Richard Hofer, Owen Kahn, Ted Fujimoto, Tom Sainsbury, Uri Laseron

If you have ideas or want to contribute, submit an issue or pull request, or reach out to us on [gitter](https://gitter.im/thunder-project/thunder), [twitter](https://twitter.com/thefreemanlab), or the [mailing list](https://groups.google.com/forum/?hl=en#!forum/thunder-user).
