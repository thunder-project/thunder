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

#### data
ica_test.txt - example data for running ICA (from [FastICA for Matlab](http://research.ics.aalto.fi/ica/fastica/code/dlcode.shtml))

#### helper
copy-numpy-scipy-ec2.sh - copy numpy and scipy to all workers on an EC2 cluster


## To-Do

rpca.py - robust PCA

cca.py - canonical correlation analysis

fourier.py - amplitude and phase (of time series)

scala versions
