[![Build Status](https://travis-ci.org/freeman-lab/thunder.png)](https://travis-ci.org/freeman-lab/thunder)

Thunder
=======

Large-scale neural data analysis with Spark

## About

Thunder is a library for analyzing large-scale neural data. It's fast to run, easy to develop for, and can be run interactively. It is built on Spark, a powerful new framework for distributed computing.

Thunder includes low-level utilties for data loading, saving, signal processing, and fitting algorithms (regression, factorization, etc.), and high-level functions that can be scripted to easily combine analyses. It is written in Spark's Python API (Pyspark), making use of scipy and numpy. We plan to port some or all functionality to Scala in the future, but for now all scala functions should be considered prototypes.

## Quick start

Here's a quick guide to getting up and running. It assumes [Scala 2.10.3](http://www.scala-lang.org/download/2.10.3.html), [Spark 0.9.0](http://spark.incubator.apache.org/downloads.html), and [Python 2.7.6](http://www.python.org/download/releases/2.7.6/) (with [NumPy](http://www.numpy.org/), [SciPy](http://scipy.org/scipylib/index.html), and [Python Imaging Library](http://www.pythonware.com/products/pil/)) are already installed. First, download the latest [build](https://github.com/freeman-lab/thunder/archive/master.zip) and add it to your path.

	PYTHONPATH=your_path_to_thunder/python/:$PYTHONPATH

Now go into the top-level Thunder directory and run an analysis on test data.

	$SPARK_HOME/bin/pyspark python/thunder/factorization/pca.py local data/iris.txt ~/results 4

This will run principal components on the “iris” data set with 4 components, and write results to a folder in your home directory. The same analysis can be run interactively in a shell. Start Pyspark:

	$SPARK_HOME/bin/pyspark

Then run the analysis

	>> from thunder.util.load import load
	>> from thunder.factorization.pca import pca
	>> data = load(sc, 'data/iris.txt')
	>> scores, latent, comps = pca(data, 4)

For running in the shell, we include a script for automatically importing commonly used functions

	>> execfile('helper/thunder-startup.py')

To run in iPython, just set

	>> export IPYTHON=1

## Analyses

Thunder currently includes four packages: clustering, factorization, regression, and signal processing, as well as utils for shared methods like loading and saving (see Input format and Output format). Individual packages include both high-level analyses and underlying methods and algorithms. There are several stand-alone analysis scripts for common analysis routines, but the same functions (or sub-functions) can be used from within the PySpark shell for easy interactive analysis. Here is a list of the primary analyses:

### clustering

_kmeans_ - k-means clustering

### factorization

_pca_ - principal components analysis  
_ica_ - independent components analysis

### regression

_regress_ - mass univariate regression (linear and bilinear)
_regresswithpca_ - regression combined with dimensionality reduction
_tuning_ - mass univariate parameteric tuning curves (circular and gaussian)

### signal processing

_crosscorr_ - signal cross-correlation  
_fourier_ - fourier analysis  
_localcorr_ - local spatial time series correlations  
_stats_ - summary statistics (mean, std, etc.)  
_query_ - average over indices  


## Input and output

Thunder is built around a commmon input format for raw data: a set of neural signals as key-value pairs, where the key is an identifier, and the value is a response time series. In imaging data, for example, each record would be a voxel, the key an xyz coordinate, and the value a flouresence time series. This is a useful and efficient representation of raw data because the analyses parallelize across neural signals (i.e. across records). 

These key-value records can, in principle, be stored in a variety of formats on a cluster-accessible file system; the core functionality (besides loading) does not depend on the file format, only that the data are key-value pairs. Currently, the loading function assumes a text file input, where the rows are neural signals, and the columns are the keys and values, each number separated by space. But we are investigating alternative file formats that are more space-efficient, as well as developing scripts that faciliate converting raw data (e.g. tif images) into the commmon data format.

All metadata (e.g. parameters of the stimulus or behavior for regression analyses) can be provided as numpy arrays or loaded from MAT files, see relavant functions for more details.

Results can be visualized directly from the python shell using matplotlib, or saved as MAT files (including automatic reshaping and sorting), text files, or images (including automatic rescaling).
