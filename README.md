[![Build Status](https://travis-ci.org/freeman-lab/thunder.png)](https://travis-ci.org/freeman-lab/thunder)

Thunder
=======

Large-scale neural data analysis with Spark

## About

Spark is a powerful new framework for cluster computing, particularly well suited to iterative computations. Thunder is a family of analyses for finding structure in neural data using machine learning algorithms. It's fast to run, easy to develop for, and can be run interactively.

Thunder includes low-level utilties for data loading, saving, signal processing, and shared algorithms (regression, factorization, etc.), and high-level functions that can be scripted to easily combine analyses. The entire package is written in Spark's Python API (Pyspark), making use of scipy and numpy. We plan to port some or all functionality to Scala in the future (e.g. for streaming), but for now all scala functions should be considered prototypes.

## Quick start

Here's a quick guide to getting up and running. It assumes [Spark](http://spark.incubator.apache.org/downloads.html), Numpy, and Scipy are already installed. First, download the latest [build](https://github.com/freeman-lab/thunder/archive/master.zip) and add it to your path.

	PYTHONPATH=your_path_to_thunder/python/:$PYTHONPATH

Now go into the top-level Thunder directory and run an analysis on test data.

	$SPARK_HOME/pyspark python/thunder/factorization/pca.py local data/iris.txt ~/results 4

This will run principal components on the “iris” data set with 4 components, and write results to a folder in your home directory. The same analysis can be run interactively in a shell. Start Pyspark:

	$SPARK_HOME/pyspark

Then run the same analysis

	>> from thunder.util.dataio import parse, saveout
	>> from thunder.factorization.pca import pca
	>> lines = sc.textFile(”data/iris.txt”)
	>> data = parse(lines).cache()
	>> scores, latent, comps = pca(data, 4)

## Analyses

Thunder currently includes four packages: clustering, factorization, regression, and signal processing, as well as a utils for shared methods like loading and saving (see Input format and Output format). Individual packages include both high-level analyses and underlying methods and algorithms. There are several stand-alone analysis scripts for common analyses, but the same functions (or sub-functions) can be used from within the Pyspark shell for easy interactive analysis.

### clustering

analyses:  
_kmeans_ - k-means clustering

### factorization

analyses:  
_pca_ - principal components analysis  
_ica_ - independent components analysis

utilities:  
singular value decomposition

### regression

analyses:  
_regress_ - regression (linear and bilinear)  
_tuning_ - parameteric tuning curves (circular and gaussian)

utilities:  
creating and fitting regression models  
creating and fitting tuning models

### signal processing

analyses:  
_crosscorr_ - signal cross-correlation  
_fourier_ - fourier analysis  
_localcorr_ - local spatial time series correlations  
_stats_ - summary statistics (mean, std, etc.)  
_query_ - average over indices  

utilities:  
creating and calculating signal processing methods


## Input formats

All functions use the same input format data format: a text file, where the rows are voxels or neurons and the columns are time points. The first three entries in each row are the x,y,z coordinates of that voxel (or some other identifier), and the subsequent entries are the signals for that voxel at each time point. For example, a data set with 2x2x2 voxels and 8 time points might look like:

	1 1 1 11 41 2 17 43 24 56 87
	1 2 1 ...
	2 1 1 ...
	2 2 1 ...
	1 1 2 ...
	1 2 2 ...
	2 1 2 ...
	2 2 2 ...

Subsets of voxels (e.g. different imaging planes) can be stored in separate text files within the same directory, or all in one file.

When parsing data, preprocessing can be applied in the form of mean subtraction, or conversion to df/f (subtract and divide by the mean)

Covariates (e.g. related to the stimulus or task, for regression analyses) can be loaded from MAT files or provided directly as numpy arrays, see appropriate functions for more details.
