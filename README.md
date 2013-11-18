Thunder
=======

Library for neural data analysis with the Spark cluster computing framework

## About

Spark is a powerful new framework for cluster computing, particularly well suited to iterative computations; see the [project webpage](http://spark-project.org/documentation.html). Thunder is a family of analyses for finding structure in high-dimensional spatiotemporal neural imaging data (e.g. calcium) implemented in Spark.

## Use

To run these functions, first [install Spark](http://spark-project.org/downloads/) and [scala](http://www.scala-lang.org/downloads).

For python functions, call using pyspark:

	SPARK_HOME/pyspark ica.py local data/ica_test.txt results 4 4

For scala functions, build and run in sbt:

	sbt package
	sbt "run local data/hierarchical_test.txt results.txt"

## Input formats

All functions rely on input neural data, and some additionally rely on information about external variables (i.e. covariates).

All functions use the same format for neural data: a text file, where the rows are pixels and the columns are time points. The first three entries in each row are the x,y,z coordinates of that pixel, and the subsequent entries are the neural signals at each time point. For example, a data set with 2x2x2 pixels and 8 time points would look like:

	1 1 1 11 41 2 17 43 24 56 87
	1 2 1 ...
	2 1 1 ...
	2 2 1 ...
	1 1 2 ...
	1 2 2 ...
	2 1 2 ...
	2 2 2 ...

Different subsets of pixels (e.g. different planes) can be stored in separate text files, or all in one file.

Many functions also use a common format for covariates: a text file of 0s and 1s, where the rows are variables, and the columns are time points. For example, if eight orientations were presented in random order for the example above, the file would be:

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

## Contents

#### python
pca - principal components analysis

empca - iterative PCA using EM algorithm

ica - independent components analysis

cca - canonical correlation analysis

rpca - robust PCA

fourier - fourier analysis on time series data

query - get average time series with desired indices

kmeans - k-means clustering

#### scala

bisecting - divisive hierarchlal clustering using bisecting k-means

hierarchical - agglomerative hierachical clustering

mantis - streaming analysis of neuroimaging data (prototype)

## To-Do

scala versions of all functions
