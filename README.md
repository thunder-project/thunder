[![Build Status](https://travis-ci.org/freeman-lab/thunder.png)](https://travis-ci.org/freeman-lab/thunder)

Thunder
=======

Large-scale neural data analysis with Spark

## About

Spark is a powerful new framework for cluster computing, particularly well suited to iterative computations. Thunder is a family of analyses for finding structure in neural data using machine learning algorithms. It's fast to run, easy to develop for, and can be run interactively.

Thunder includes low-level utilties for data loading, saving, signal processing, and shared algorithms (regression, factorization, etc.), and high-level functions that can be scripted to easily combine analyses. The entire package is written in Spark's Python API (Pyspark), making use of scipy and numpy. We plan to port some or all functionality to Scala in the future (e.g. for streaming), but for now all scala functions should be considered prototypes.

## Quick start

Here's a quick guide to getting up and running. It assumes [Scala 2.9.3](http://www.scala-lang.org/download/2.9.3.html), [Spark 0.8.1](http://spark.incubator.apache.org/downloads.html), and [Python 2.7.6](http://www.python.org/download/releases/2.7.6/) (with [NumPy](http://www.numpy.org/), [SciPy](http://scipy.org/scipylib/index.html), and [Python Imaging Library](http://www.pythonware.com/products/pil/)) are already installed. First, download the latest [build](https://github.com/freeman-lab/thunder/archive/master.zip) and add it to your path.

	PYTHONPATH=your_path_to_thunder/python/:$PYTHONPATH

Now go into the top-level Thunder directory and run an analysis on test data.

	$SPARK_HOME/pyspark python/thunder/factorization/pca.py local data/iris.txt ~/results 4

This will run principal components on the “iris” data set with 4 components, and write results to a folder in your home directory. The same analysis can be run interactively in a shell. Start Pyspark:

	$SPARK_HOME/pyspark

Then run the same analysis

	>> from thunder.util.parse import parse
	>> from thunder.factorization.pca import pca
	>> lines = sc.textFile(”data/iris.txt”)
	>> data = parse(lines).cache()
	>> scores, latent, comps = pca(data, 4)

## Analyses

Thunder currently includes four packages: clustering, factorization, regression, and signal processing, as well as a utils for shared methods like loading and saving (see Input format and Output format). Individual packages include both high-level analyses and underlying methods and algorithms. There are several stand-alone analysis scripts for common analyses, but the same functions (or sub-functions) can be used from within the Pyspark shell for easy interactive analysis. Here is a list of the primary analyses:

### clustering

_kmeans_ - k-means clustering

### factorization

_pca_ - principal components analysis  
_ica_ - independent components analysis

### regression

_regress_ - regression (linear and bilinear)  
_tuning_ - parameteric tuning curves (circular and gaussian)

### signal processing

_crosscorr_ - signal cross-correlation  
_fourier_ - fourier analysis  
_localcorr_ - local spatial time series correlations  
_stats_ - summary statistics (mean, std, etc.)  
_query_ - average over indices  


## Input and output

All functions use the same format for primary input data: a text file, where the rows are neural signals (e.g. voxels, neurons) and the columns are time points. The first entries in each row are optional key identifiers (e.g. the x,y,z coordinates of each voxel), and subsequent entries are the response values for that signal at each time point (e.g. calcium flouresence, spike counts). For example, an imaging data set with 2x2x2 voxels and 8 time points might look like:

	1 1 1 11 41 2 17 43 24 56 87
	1 2 1 ...
	2 1 1 ...
	2 2 1 ...
	1 1 2 ...
	1 2 2 ...
	2 1 2 ...
	2 2 2 ...

Subsets of neural signals (e.g. from different imaging planes) can be stored in separate text files within the same directory, or all in one file. Covariates (e.g. related to the stimulus or task, for regression analyses) can be loaded from MAT files or provided directly as numpy arrays, see appropriate functions for more details.

When parsing data, preprocessing can be applied to each neural signal (e.g. conversion to dF/F for imaging data).

Results can be saved as MAT files, text files, or images (including automatic rescaling).
