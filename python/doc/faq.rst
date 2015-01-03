.. _contributing:

.. currentmodule:: thunder

F.A.Q.
======

Basic usage
~~~~~~~~~~~

*When should I cache my data?*
	
	Any data object in Thunder can be cached into RAM by calling ``data.cache()`` on it. Caching is a powerful capability of Spark, and can greatly speed up subsequent queries or operations on the data. However, you can only cache as much data as your cluster has RAM (and the actual RAM avaialble to Spark for caching is approximately half of the total installed RAM of the cluster). It often makes sense to cache intermediate results, for example, perform filtering on a set of images, convert to a series, and cache the result. You should rarely cache data you only plan to use once; conversely, you should almost always cache data that will be subjected to iterative algorithms (like clustering).


*I performed an operation and it happened instantaneously, is it really that fast?*

	Probably not. Many distributed operations in Thunder and Spark are lazy, which means they will not be executed until the neccessary output requires them to be. For example, if you call ``images.medianFilter().sum()`` on an ``Images`` object, the first filtering operation is lazy, and is computed only because you asked for the sum. If you had just called ``imgs.medianFilter()`` it would appear to have happened immediately, but nothing would have actually been computed (yet).


*Can I index directly into an Images or Series object?*

	Not exactly. You can use ``filterOnKey`` to select one or more entries from a data object, which should be very fast on memory-cached data. For example, if your data have tuples as keys, calling ``data.filterOnKeys(lambda k: k == (1,10,2))`` will return the record with key ``(1,10,2)``, but this still requires a scan through the data. We are hoping to provide faster direct indexing support in the future.


*I’m routinely converting my Images data to Series and only ever working with the Series data, can I avoid this conversion step?*

	Yes, you can! You just have to save your Images data to binary Series data. This requires storing an extra copy of the data (bad), but will make it faster to load data in the future (good). The ``ThunderContext`` provides a method for this conversion, ``convertImagesToSeries``, which takes as arguments an input path for the Images data and an output path (typically either a networked file system volume or AWS S3 bucket) to which to write the Series output. The Series data can subsequently be loaded via the ``tsc.loadSeries`` method. Alternatively, one could load the image data by ``tsc.loadImages``, and then save it with the ``saveAsBinarySeries`` method. 


*Is there sample data I can try analyses on?*
	
	Yes, there are currently sample data sets of light-sheet imaging data available (part of a collaboration with Misha Ahren's lab). They cannot be directly downloaded, but are available through the ``ThunderContext.loadExampleEC2`` method.


*How should I visualize the results of my analysis?*
	
	We recommend running Thunder in an iPython notebook. For simple plots, you can use ``matplotlib``, along with the excellent library ``seaborn``, to look at the output of many analyses directly in the notebook. We are currently developing a separate library Lightning for interactive visualization, that will work within the notebook. It is still early in development, but check it out!


*I’m running Thunder locally and it doesn’t seem particularly fast, why is that?*
	
	Thunder is designed to run on a cluster. One nice feature is that you can run things locally, which is a great way to debug. But local usage may not be particularly fast, and for very small data sets (that can be loaded entirely on a single machine), for some operations, it may be faster to use local operations. We are investigating ways to speed up these workflows.


*I just did a series.pack() and the array I get back is all screwed up!*
	
	Try specifying ``sorting=True`` in the ``pack()`` call. The most likely problem is the ordering of your data. ``Series.pack()`` assumes by default that its keys are in "series-sorted" order, with the ``x`` dimension changing most rapidly, followed by ``y``, and finally ``z``. If you create series data within Thunder (e.g. from an ``Images`` object), it should be in this order already. But if you created the data yourself, or under some other conditions, it may have the wrong order and need sorting. 


*When trying to work with binary Series data, I get an error similar to the following: "java.lang.IncompatibleClassChangeError: Found interface org.apache.hadoop.mapreduce.JobContext, but class was expected"*
	
	This error indicates that the Spark Hadoop version that Thunder was compiled against doesn’t match the version that is actually installed on the cluster. The jar files distributed with Thunder by default are compiled against Hadoop 1.0.4, which is what you will get with any of the “Pre-built for Hadoop 1.x” precompiled Spark distributions available from https://spark.apache.org/downloads.html. If Hadoop 2.x jars are found on the classpath at runtime rather than Hadoop 1.x jars, then you will see this error anytime the Thunder jar is used. The simplest workaround would be to make sure that ``$SPARK_HOME`` refers to a version of Spark compiled against Hadoop 1.x. If you prefer to use a version of Spark compiled for Hadoop 2.x, then at present this requires Thunder to be rebuilt from source. (See instructions elsewhere).

Configuration and installation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

My university / research institute has a private cluster, can I install Spark on it?
Yes, but won't go into detail here

How many cluster nodes / what total amount of RAM do I need?

What configuration parameters should I use?


Using Thunder on Amazon's EC2
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

How expensive is S3 and EC2?

Where should I store my data?

What EC2 instance types should I use?

I’m concerned about the costs of running a large EC2 cluster. How can I save money?

I see repeated errors like the following when trying to stand up a cluster on Amazon EC2 using the thunder-ec2 script - what should I do?





