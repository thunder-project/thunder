[![Build Status](https://travis-ci.org/freeman-lab/thunder.png)](https://travis-ci.org/freeman-lab/thunder)

Thunder
=======

Large-scale neural data analysis with Spark - <http://freeman-lab.github.io/thunder/>

## About

Thunder is a library for analyzing large-scale neural data. It's fast to run, easy to develop for, and can be run interactively. It is built on Spark, a powerful new framework for distributed computing.

Thunder includes utilties for data loading and saving, and modular functions for time series statistics, matrix decompositions, and fitting algorithms. Analyses can easily be scripted or combined. It is written in Spark's Python API (Pyspark), making use of scipy, numpy, and scikit-learn. Experimental streaming analyses are availiable in Scala, and we plan to port some or all functionality to Scala in the future.

## Quick start

Here's a quick guide to getting up and running. If you are already running Spark on a cluster, you should have all the dependencies you need. If you are running Spark locally, we assume [Scala 2.10.3](http://www.scala-lang.org/download/2.10.3.html), [Spark 1.0.0](http://spark.incubator.apache.org/downloads.html), and [Python 2.7.6](http://www.python.org/download/releases/2.7.6/) (with [NumPy](http://www.numpy.org/), [SciPy](http://scipy.org/scipylib/index.html), [Scikit learn](http://scikit-learn.org/stable/) and [Python Imaging Library](http://www.pythonware.com/products/pil/)) are already installed. First, download the latest [build](https://github.com/freeman-lab/thunder/archive/master.zip) and add it to your path.

	PYTHONPATH=<your/path/to/thunder>/python/:$PYTHONPATH

Now run an analysis on test data.

	<your/path/to/spark>/bin/spark-submit <your/path/to/thunder>/python/thunder/factorization/pca.py local data/iris.txt ~/results 4

This will run principal components on the “iris” data set with 4 components, and write results to a folder in your home directory. The same analysis can be run interactively. Start PySpark:

	<your/path/to/spark>/bin/pyspark

Then do the analysis

	>> from thunder.io import load
	>> from thunder.factorization import PCA
	>> data = load(sc, 'data/iris.txt')
	>> pca = PCA(k=4)
	>> pca.fit(data)

To run in iPython, just set this environmental variable before staring PySpark:

	export IPYTHON=1

If you are running Thunder on a cluster, create an egg first:

	cd <your/path/to/thunder>/python/
	./setup.py bdist_egg

We include a script for automatically importing commonly used functions in the shell

	>> execfile('helper/thunder-startup.py')

Finally, we include a script for easily launching an Amazon EC2 cluster with Thunder presintalled

	>> <your/path/to/thunder>/helper/ec2/thunder-ec2 -k mykey -i mykey.pem -s <number-of-nodes> launch <cluster-name>

## Analyses

Thunder currently includes five packages: classification, clustering, factorization, regression, and timeseries, as well as an io package for loading and saving (see Input format and Output format), and a util package for utilities (like common matrix operations). Packages include scripts for running standalone analyses, but the underlying classes and functions can be used from within the PySpark shell for easy interactive analysis. Here is a list of the primary analyses:

### classification

_classify_ - mass univariate classifiaction

### clustering

_kmeans_ - k-means clustering

### factorization

_pca_ - principal components analysis  
_ica_ - independent components analysis
_svd_ - singular value decomposition

### regression

_regress_ - mass univariate regression (linear and bilinear)  
_regresswithpca_ - regression combined with dimensionality reduction  
_tuning_ - mass univariate parameteric tuning curves (circular and gaussian)  

### timeseries

_crosscorr_ - time series cross-correlation  
_fourier_ - fourier analysis  
_localcorr_ - local spatial time series correlations  
_stats_ - time series statistics (mean, std, etc.)  
_query_ - query time series data by averaging over subsets


## Input and output

Thunder is built around a commmon input format for raw neural data: a set of signals as key-value pairs, where the key is an identifier, and the value is a response time series. In imaging data, for example, each record would be a voxel or an ROI, the key an xyz coordinate, and the value a flouresence time series. This is a useful and efficient representation of raw data because the analyses parallelize across neural signals (i.e. across records). 

These key-value records can, in principle, be stored in a variety of formats on a cluster-accessible file system; the core functionality (besides loading) does not depend on the file format, only that the data are key-value pairs. Currently, the loading function assumes a text file input, where the rows are neural signals, and the columns are the keys and values, each number separated by space. Support for flat binary files is coming soon. We are also developing scripts that faciliate converting raw data (e.g. images) into the commmon data format.

All metadata (e.g. parameters of the stimulus or behavior for regression analyses) can be provided as numpy arrays or loaded from MAT files, see relavant functions for more details.

Results can be visualized directly from the python shell using matplotlib, or saved as MAT files (including automatic reshaping and sorting), text files, or images (including automatic rescaling).
