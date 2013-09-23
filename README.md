Thunder
=======

Library for neural data analysis with the Spark cluster computing framework

## About

Spark is a powerful new framework for cluster computing, particularly well suited to iterative computations; see the [project webpage](http://spark-project.org/documentation.html). Thunder is a collection of model-fitting routines for analyzing high-dimensional spatiotemporal neural data implemented in Spark.

## Use

To run these functions, first [install Spark](http://spark-project.org/downloads/) and [scala](http://www.scala-lang.org/downloads).

For python functions, call using pyspark:

	SPARK_HOME/pyspark ica.py local data/ica_test.txt results 4 4

For scala functions, build and run in sbt:

	sbt package
	sbt "run local data/hierarchical_test.txt results.txt"

If running on a cluster (e.g. Amazon's EC2), numpy, and any other dependencies, must be installed on all workers. See the helper scripts for doing this on EC2.

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

mantis - streaming analysis of neuroimaging data

## To-Do

scala versions of all functions
