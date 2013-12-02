Thunder
=======

Library for neural data analysis with the Spark cluster computing framework

## About

Spark is a powerful new framework for cluster computing, particularly well suited to iterative computations; see the [project webpage](http://spark-project.org/documentation.html). Thunder is a family of analyses for finding structure in high-dimensional spatiotemporal neural imaging data (e.g. calcium) implemented in Spark. 

It includes low-level utilties for data loading, saving, signal processing, and shared algorithms (regression, factorization, etc.), and high-level functions that can be scripted to easily combine analyses. The standard package is written in Python with Pyspark, making extensive use of scipy and numpy. A subset of functions, including  prototypes of real-time analyses, are availaible for Scala., because they use functionality not yet availiable in Pyspark. We plan to port all functionality to Scala in the future.

## Use

To run these functions, first [install Spark](http://spark-project.org/downloads/) and [scala](http://www.scala-lang.org/downloads).

For python functions, call individual functions using pyspark:

	pyspark pca.py local data/iris.txt results 4

Or run iteractively in the Pyspark shell

	pyspark
	>> lines = sc.textFile("data/iris.txt")
	>> data = parse(lines).cache()
	>> comps,latent,scores = svd1(data,3)

For scala functions, build and run in sbt:

	sbt package
	sbt "run local data/hierarchical_test.txt results.txt"

## Input formats

All functions use neural data as input, and some additionally use information about external covariates (e.g. stimuli or behavioral attributes).

All functions use the same format for neural data: a text file, where the rows are voxels and the columns are time points. The first three entries in each row are the x,y,z coordinates of that voxel, and the subsequent entries are the neural signals for that voxel at each time point. For example, a data set with 2x2x2 voxels and 8 time points might look like:

	1 1 1 11 41 2 17 43 24 56 87
	1 2 1 ...
	2 1 1 ...
	2 2 1 ...
	1 1 2 ...
	1 2 2 ...
	2 1 2 ...
	2 2 2 ...

Subsets of voxels (e.g. different imaging planes) can be stored in separate text files within the same directory, or all in one file.

Many functions make use of covariates, and there is a common input format: a text file of 0s and 1s, where the rows are variables, and the columns are time points. For example, if eight orientations were presented in random order for the example above, the file would be:

	1 0 0 0 0 0 0 0
	0 0 1 0 0 0 0 0
	0 0 0 0 1 0 0 0
	0 1 0 0 0 0 0 0
	0 0 0 0 0 0 1 0
	0 0 0 0 0 0 0 1
	0 0 0 0 0 1 0 0
	0 0 0 1 0 1 0 0

For parameteric models (e.g. tuning), also provide a text file with the stimulus value corresponding to each row, like this:

	0 45 90 135 180 225 270 315

## Contents (Python)

### clustering

kmeans - k-means clustering

### factorization

pca - principal components analysis

ica - independent components analysis

rpca - robust pca

### regression

regress - simple variants of linear regression

shotgun - parallel L1 regularized regression

tuning - fit parametric models

### summary

ref - compute summary statistics (mean, median, etc.)

localcorr - local spatial time series correlations

fourier - fourier analysis

query - average time series data for entries with the provided indices

## Contents (Scala)

bisecting - divisive hierarchlal clustering using bisecting k-means

kmeansOnline - online clustering using kmeans

mantis - online regression

