
.. \_basic\_usage\_tutorial:

.. currentmodule:: thunder

Thunder context
===============

The ``ThunderContext`` is the entry point for most analyses.

Construction
------------

A ``ThunderContext`` (you'll only need one) is automatically provided as
the variable ``tsc`` when you start the interactive shell using the
command line call ``thunder``. It also be created manually, in two
different ways, which can be useful when writing standalone analysis
scripts (see examples in ``thunder.standalone``). First, it can be
created from an existing instance of a ``SparkContext``:

.. code:: python

    from thunder import ThunderContext
    tsc = ThunderContext(sc)
Or it can be created directly using the same arguments provided to a
``SparkContext`` (if you execute this line in this notebook you will get
an error because you cannot run multiple ``SparkContexts`` at once):

.. code:: python

    tsc = ThunderContext.start(appName='myapp')
Loading data
------------

The primary methods for loading data are ``loadSeries`` and
``loadImages``, for loading a ``Series`` or ``Images`` object,
respectively. Here we show example syntax for loading two example data
sets included with ``thunder``, and in each case inspect the first
element. (To use these example data sets, we'll first figure out their
path on our system.) See the Input Format tutorial for more information
on loading and data types.

.. code:: python

    import os.path as pth
    datapath = pth.join(pth.dirname(pth.realpath(thunder.__file__)), 'utils/data/')
.. code:: python

    data = tsc.loadImages(datapath + '/fish/tif-stack/', inputformat='tif-stack', startidx=0, stopidx=10)
.. code:: python

    %matplotlib inline
.. code:: python

    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set_style('white')
    sns.set_context('notebook')
    plt.imshow(data.first()[1][:,:,0], cmap="gray");


.. image:: thunder_context_files/thunder_context_11_0.png


.. code:: python

    data = tsc.loadSeries(datapath + '/iris/iris.bin', inputformat='binary')
    data.first()



.. parsed-literal::

    ((0,), array([ 5.1,  3.5,  1.4,  0.2]))



Currently, ``loadImages`` can load ``tif``, ``tif-stack`` (multipage
tif), ``png``, or ``binary`` images (or volumes) from a local file
system, networked file system, or Amazon S3. ``loadSeries`` can load
data from one or more ``text`` or ``binary`` files on a local file
system, networked file system, S3, or HDFS.

There is also a method for easily loading ``Series`` data stored in
local arrays in either numpy ``npy`` or Matlab ``MAT`` format (if
loading from a ``MAT`` file, you must additionally provide a variable
name). This is not particularly useful when working with large data
sets, but may be convenient for local testing, or for distributing a
smaller data set for performing intensive computations. In the latter
case, the number of partitions should be set to be approximately equal
to 2-3 times the number of cores available on your cluster, so that
different cores can work on different portions of the data.

.. code:: python

    data = tsc.loadSeriesLocal(datapath + '/iris/iris.mat', inputformat='mat', varname='data', minPartitions=5)
    data.first()



.. parsed-literal::

    (0, array([ 5.1,  3.5,  1.4,  0.2]))



.. code:: python

    data = tsc.loadSeriesLocal(datapath + '/iris/iris.npy', inputformat='npy', minPartitions=5)
    data.first()



.. parsed-literal::

    (0, array([ 5.1,  3.5,  1.4,  0.2]))



Finally, there are two methods for performing the common operation of
converting ``Images`` data into ``Series`` data. As an example, if we
have a sequence of images that represent successive time points, we
might want to convert these to ``Series`` data where each record is the
time series of a voxel. These two methods take a set of image files as
input, and either load them directly as a ``Series``, or save the
contents out as ``binary`` data suitable for loading with
``loadSeries``.

The ordering of points in the ``Series`` values comes from the
lexicographic ordering of the image data file file names. Given image
files that each represent a single time point, indicating the time point
in the image file name so that alphabetically later file names
correspond to temporally later images (for instance by adding a suffix
'\_tp000', '\_tp001' and so on to a common file name) will result in
correct ordering of the resulting ``Series`` data.

There are two implementations of the ``Image`` to ``Series`` conversion.
Which to use is determined by the ``shuffle`` argument. If
``shuffle=True`` it will do the conversion using a distributed transpose
operation, which requires shuffling blocks of data across the cluster.
If ``shuffle=False`` it will use an alternate method that avoids the
shuffle at the cost of increased file IO. In practice, ``shuffle=False``
will often be slower, but we have found it to be more robust, especially
on larger data sets. ``shuffle=False`` is the current default.

.. code:: python

    data = tsc.loadImagesAsSeries(datapath + '/fish/tif-stack/', inputformat='tif-stack', shuffle=False)
.. code:: python

    tsc.convertImagesToSeries(datapath + '/fish/tif-stack/', 'savelocation', inputformat='tif-stack', shuffle=False)
    data = tsc.loadSeries('savelocation')
Loading examples
----------------

The ``makeExample`` method makes it easy to generate example data for
testing purposes, by calling methods from the ``DataSets`` class:

.. code:: python

    data = tsc.makeExample('kmeans', k=2, ndims=10, nrecords=10, noise=0.5)
.. code:: python

    from numpy import asarray
    sns.set_style('darkgrid')
    sns.set_context('notebook')
    ts = data.values().collect()
    plt.plot(asarray(ts).T);


.. image:: thunder_context_files/thunder_context_24_0.png


The 'loadExample' method directly loads one of the small example
datasets. This are highly compressed and downsampled, and meant only to
demonstrate basic functionality and help explore the API, not to
represent anything meaningful about the data itself.

.. code:: python

    data = tsc.loadExample('fish-series')
    img = data.seriesMean().pack()
.. code:: python

    sns.set_style('white')
    sns.set_context('notebook')
    plt.imshow(img[:,:,0].T, cmap="gray");


.. image:: thunder_context_files/thunder_context_27_0.png


Example data stored on S3 can be loaded using the ``loadExampleEC2``
method. You must be running Thunder on an Amazon EC2 cluster in order to
use these methods. Some of the available data sets include parameters
that are returnded alongside the data.

.. code:: python

    data, params = tsc.loadExampleEC2('zebrafish-optomotor-response')