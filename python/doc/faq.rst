.. _contributing:

.. currentmodule:: thunder

F.A.Q.
======

Basic usage
~~~~~~~~~~~

*When should I cache my data?*
	
	Any data object can be cached into RAM by calling ``data.cache()`` on it. Caching is a powerful capability of Spark, and can greatly speed up subsequent queries or operations on the data. However, you can only cache as much data as your cluster has RAM (and the actual RAM available to Spark for caching is approximately half of the total installed RAM of the cluster). It often makes sense to cache intermediate results, for example, perform filtering on a set of images, convert to a series, and cache the result. You should rarely cache data you only plan to use once; conversely, you should almost always cache data that will be subjected to iterative algorithms (like clustering).


*I performed an operation and it happened instantaneously, is it really that fast?*

	Probably not. Many distributed operations in Thunder and Spark are lazy, which means they will not be executed until the necessary output requires them to be. For example, if you call ``images.medianFilter().sum()`` on an ``Images`` object, the first filtering operation is lazy, and is computed only because you asked for the sum. If you had just called ``imgs.medianFilter()`` it would appear to have happened immediately, but nothing would have actually been computed (yet).


*Can I index directly into an Images or Series object?*

	Not exactly. You can use ``filterOnKey`` to select one or more entries from a data object, which should be very fast on memory-cached data. For example, if your data have tuples as keys, calling ``data.filterOnKeys(lambda k: k == (1,10,2))`` will return the record with key ``(1,10,2)``, but this still requires a scan through the data. We are hoping to provide faster direct indexing support in the future.


*I’m routinely converting my Images data to Series and only ever working with the Series data, can I avoid this conversion step?*

	Yes, you can! You just have to save your Images data to binary Series data. This requires storing an extra copy of the data (bad), but will make it faster to load data in the future (good). The ``ThunderContext`` provides a method for this conversion, ``convertImagesToSeries``, which takes as arguments an input path for the Images data and an output path (typically either a networked file system volume or AWS S3 bucket) to which to write the Series output. The Series data can subsequently be loaded via the ``tsc.loadSeries`` method. Alternatively, one could load the image data by ``tsc.loadImages``, and then save it with the ``saveAsBinarySeries`` method. 


*Is there sample data I can try analyses on?*
	
	Yes, there are currently sample data sets of light-sheet imaging data available (as part of a collaboration with Misha Ahren's lab). They cannot be directly downloaded, but are available through the ``ThunderContext.loadExampleEC2`` method.


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

*My university / research institute has a private cluster, can I install Spark on it?*

	Yes. We are preparing a blog post that explains exactly how to do this.

*How many cores / cluster nodes / total amount of RAM do I need?*

	In general, fewer nodes with more cores and RAM is better than more, under-powered nodes, because it minimizes network communication. For RAM, a good rule of thumb is to determine the size of your data (at least the portion you want to cache), and then use a cluster with twice that much RAM in total. For nodes and cores, you want approximately 2-3 cores for each partition of data, which will usually be 50-100MB of the data set.

*What configuration parameters should I use?*

	Spark has many configurable parameters (see the link). The following settings have generally proven useful when running Thunder, and are automatically set by Thunder's EC2 scripts:

	.. code-block:: python 

		spark.akka.frameSize=10000
		spark.kryoserializer.buffer.max.mb=1024
		spark.driver.maxResultSize=0
		export SPARK_DRIVER_MEMORY=20g


Using Thunder on Amazon's EC2
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

*How expensive is S3 and EC2?*
	
	For current on-demand pricing by instance type, see: http://aws.amazon.com/ec2/pricing/. For details on instance types, see: http://aws.amazon.com/ec2/instance-types/, and in particular the instance types matrix at the bottom of that page. For details on storage charges, see: http://aws.amazon.com/s3/pricing/.

*Where should I store my data?*
	
	Options are S3 and glacier.

*I want to share my data, what settings should I use?*
	
	You can make it public, but then anyone can download it, and you'll pay the bill. So make it requester pays. This requires some things to be set though.

*What EC2 instance types should I use?*
	
	Thunder workflows may be bottlenecked either by I/O or by computational limitations, depending on the nature of the workflow. Smaller EC2 instances have both lesser compute capacity and more modest network bandwidth between cluster nodes. Provisioning the same total number of cores spread across a cluster of smaller instances also increases the amount of network traffic required, relative to a cluster composed of fewer but larger instances. Finally, EC2 pricing is generally roughly equivalent for the same total amount of memory and CPUs across a cluster, no matter whether these resources are concentrated in a few large instances or spread across many smaller ones. Overall, this argues for using fewer, larger instances. With Spark, I/O limits can often be effectively addressed by caching in memory (after an initial read from disk or S3), suggesting that high-memory (“r”-type) instances may be a good choice for I/O-bound workflows. Conversely, compute-optimized (“c”-type) instances are likely most cost effective for computationally-intensive workflows. Both these instance types also have 10GB networking available for the largest instances. The thunder-ec2 script currently defaults to m3.2xlarge instances, which represent a balance between memory and compute capacity. 

*I’m concerned about the costs of running a large EC2 cluster. How can I save money?*

	Consider using spot instances: http://aws.amazon.com/ec2/purchasing-options/spot-instances/. These allow you to set a maximum hourly price that you would be willing to pay for an instance. You will be charged at the spot instance rate for your particular availability zone and instance type, which reflects the current available capacity at the AWS data centers and which will typically be much less than the standard on-demand rate. The downside is that if the spot rate ever exceeds your bid, then you will lose the instances; thus, spot instances are not recommended for last-minute analyses before an abstract submission deadline or similar. However, in practice, spot instances can be very long-lived, and represent a practical and affordable alternative for most data analysis tasks. 

*I'm getting a message saying that I've requested too many instances*

	By default, AWS will limit the maximum number of instances or spot instances of a given type that can be running under a single account. The high-memory (“r”-type) nodes in particular have relatively low spot instance limits: http://docs.aws.amazon.com/AWSEC2/latest/UserGuide/using-spot-limits.html. 


*I see repeated errors when starting a cluster on Amazon EC2, what should I do?*

	You will most likely see the following error:

	.. code-block:: python 

		ssh: connect to host ec2-54-197-68-74.compute-1.amazonaws.com port 22: Connection refused
		Error 255 while executing remote command, retrying after 30 seconds

	This probably indicates that for whatever reason, one or more of the EC2 nodes has been slow to start. Oftentimes this will resolve itself; after waiting ~5 minutes, try resuming the cluster launch, by adding the --resume flag to the same thunder-ec2 command line as previously. Alternatively, you can simply terminate the existing cluster and try again. This most likely represents a transient AWS issue, rather than a problem with Thunder or Spark per se. This issue appears to be much reduced using Spark 1.2, which will automatically wait for all instances to be in an “ssh-ready” state before attempting to connect. If you encounter this problem repeatedly, consider updating the installation of Spark on the machine you use to launch the EC2 cluster to Spark 1.2.0 or later. 






