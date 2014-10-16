
.. \_clustering\_tutorial:

.. currentmodule:: thunder

Clustering
==========

KMeans clustering is a simple way to explore structure in time series
data, by finding groups of signals with similar response properties.
Here, we show how to use clustering and how to inspect the results in
the form of a spatial map.

Inspect the data
----------------

.. code:: python

    # load example data, and cache it to speed up repeated queries
    data = tsc.loadExample('fish-series').normalize()
    data.cache()
    data.dims;
.. code:: python

    %matplotlib inline
.. code:: python

    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set_context('notebook')
Extract some samples just to look at the typical structure of the time
series:

.. code:: python

    examples = data.subset(nsamples=50, thresh=0.1)
    sns.set_style('darkgrid')
    sns.set_palette('muted')
    plt.plot(examples.T);


.. image:: clustering_files/clustering_8_0.png


Cluster
-------

Perform KMeans clustering with 10 clusters:

.. code:: python

    from thunder import KMeans
    model = KMeans(k=10).fit(data)
Plot the time series associated with each of the recovered clusters,
with a fixed but arbitrary color scheme:

.. code:: python

    from matplotlib.colors import ListedColormap
    sns.set_style('darkgrid')
    cmap_cat = ListedColormap(sns.color_palette("hls", 10), name='from_list')
    plt.gca().set_color_cycle(cmap_cat.colors)
    plt.plot(model.centers.T);


.. image:: clustering_files/clustering_12_0.png


Compute a predicted label for each pixel, and collect it as an image of
labels:

.. code:: python

    labels = model.predict(data)
    img_labels = labels.pack()
Look at the resulting map as an image, using the same color scheme as
for plotting the cluster centers above:

.. code:: python

    sns.set_style('white')
    plt.imshow(img_labels[:,:,0].T, cmap=cmap_cat);


.. image:: clustering_files/clustering_16_0.png


The same colorization can be performed using Thunder's ``Colorize``
class. In this case the effect is the same as setting the color map in
matplotlib's ``imshow``, but later ``Colorize`` will let us perform more
complex operations.

.. code:: python

    from thunder import Colorize
    brainmap = Colorize(totype=cmap_cat).images(img_labels[:,:,0].T)
    plt.imshow(brainmap);


.. image:: clustering_files/clustering_18_0.png


Smarter color selection
-----------------------

Remember that the color assignments we used above were essentially an
arbitrary mapping from cluster center to color. When we do clustering,
however, it is often the case that some centers are more similar to one
another, and it can be easier to interpret the results if the colors are
choosen based on these relative similarities. The ``Colorize`` method
``optimize`` tries to find a set of colors such that similaries among
colors match similaries among an input array (in this case, the cluster
centers). The optimization is non-unique, so you can run multiple times
to generate different color schemes.

.. code:: python

    newclrs = Colorize.optimize(model.centers, ascmap=True)
Note that centers that resemble one another have similar colors:

.. code:: python

    sns.set_style('darkgrid')
    plt.gca().set_color_cycle(newclrs.colors)
    plt.plot(model.centers.T);


.. image:: clustering_files/clustering_22_0.png


And if we look at the map, we now see that similar regions are colored
similarly (e.g. top and bottom), which makes the spatial organization
more clear.

.. code:: python

    sns.set_style('white')
    brainmap = Colorize(totype=newclrs).images(img_labels[:,:,0].T)
    plt.imshow(brainmap);


.. image:: clustering_files/clustering_24_0.png


Thresholding
------------

One problem with what we've done so far is that clustering was performed
on all pixels, but many of them were purely noise (e.g. those outside
the brain), and some of the resulting clusters capture these noise
signals. A simple trick is to perform clustering after first
subselecting pixels based on the standard deviation of their time
series. First, let's look at a map of the standard deviation, to find a
reasonable threshold that preserves most of the relavant signal, but
ignores the noise.

.. code:: python

    std_map = data.seriesStdev().pack()
    sns.set_style('white')
    # try different values of the threshold
    plt.imshow(std_map[:,:,0].T > 0.05);


.. image:: clustering_files/clustering_26_0.png


Now perform clustering again after filtering the data based on standard
deviation

.. code:: python

    from numpy import std
    filtered = data.filterOnValues(lambda x: std(x) > 0.05)
    model = KMeans(k=10).fit(filtered)
.. code:: python

    newclrs = Colorize.optimize(model.centers, ascmap=True)
    sns.set_style('darkgrid')
    plt.gca().set_color_cycle(newclrs.colors)
    plt.plot(model.centers.T);


.. image:: clustering_files/clustering_29_0.png


.. code:: python

    img_labels = model.predict(data).pack()
    brainmap = Colorize(totype=newclrs).images(img_labels[:,:,0].T)
    sns.set_style('white')
    plt.imshow(brainmap);


.. image:: clustering_files/clustering_30_0.png


Adding similarity
-----------------

These maps are slightly odd because pixels that did not survive our
threshold still end up colored as something. A final useful trick is to
mask pixels based on how well they match the cluster they belong to. We
can compute this using the ``similarity`` method of ``KMeansModel``.

.. code:: python

    sim = model.similarity(data)
    img_sim = sim.pack()
.. code:: python

    plt.imshow(img_sim[:,:,0].T, cmap='gray',clim=(0,1));


.. image:: clustering_files/clustering_33_0.png


We can then use this as a linear mask on the colorization output

.. code:: python

    brainmap = Colorize(totype=newclrs).images(img_labels[:,:,0].T, mask=img_sim[:,:,0].T)
    plt.imshow(brainmap);


.. image:: clustering_files/clustering_35_0.png


So far we've always been colorizing one plane, but we can apply the same
operation to the full 3D volume (which in this case has just two
planes), and then look at a maximum projection.

.. code:: python

    brainmap = Colorize(totype=newclrs).images(img_labels, mask=img_sim)
.. code:: python

    from numpy import amax
    plt.imshow(amax(brainmap,2).transpose(1,0,2));


.. image:: clustering_files/clustering_38_0.png

