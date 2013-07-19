Thunder
=======

Library for neural data analysis with the Spark cluster computing framework

## About

Spark is a powerful new framework for cluster computing, particularly well suited to iterative computations; see the [project webpage](http://spark-project.org/documentation.html). Thunder is a collection of model-fitting routines for analyzing high-dimensional spatiotemporal neural data implemented in Spark.

## Use

To run these functions, first [install Spark](http://spark-project.org/downloads/) and [scala](http://www.scala-lang.org/downloads), then call like this:

SPARK_HOME/pyspark ica.py local data/ica_test.txt test 4 4

If running on a cluster (e.g. Amazon's EC2), numpy, and any other dependencies, must be installed on all workers. See the helper scripts for doing this.

## Contents

#### main
pca.py - PCA on a data matrix, e.g. space x time

ica.py - ICA on a data matrix, e.g. space x time

cca.py - CCA on a data matrix, e.g. space x time

rpca.py - robust PCA on a data matrix, e.g. space x time

fourier.py - fourier analysis on a time series matrix


#### data
ica_test.txt - example data for running ICA (from [FastICA for Matlab](http://research.ics.aalto.fi/ica/fastica/code/dlcode.shtml))

fourier_test.txt - example signals for fourier analysis

rpca_test.txt - example input matrices for rpca

cca_test.txt - example input matrices for cca


## To-Do

PCA with EM

scala versions
