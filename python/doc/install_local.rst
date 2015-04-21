.. _install_local_ref:

Installation
============

Thunder can be installed on most modern computers running Python 2, version 2.6 or higher. It has primarily been tested on Mac OS X, but should also work on Windows and Linux. If you aren't already using scientific Python, we recommend getting the `Anaconda distribution <https://store.continuum.io/cshop/anaconda/>`_.

To follow along with the instructions below, you'll just need a command line (e.g. the Terminal app in Mac OS X), a web browser, and the ``pip`` Python package manager (which is included with Anaconda). These instructions get Spark and Thunder running on a local machine. That's great for learning the library and developing analyses. But Thunder is meant to run on a cluster. Once you install, the command ``thunder-ec2`` lets you launch a cluster on Amazon's EC2, see :ref:`install_ec2_ref`.

Installing Spark 
~~~~~~~~~~~~~~~~
First you need a working installation of Spark. We strongly recommend `downloading <http://spark.apache.org/downloads.html>`_ a pre-built version (pick the one labeled Hadoop 1.X). Follow this `direct download <http://d3kbcqa49mib13.cloudfront.net/spark-1.3.0-bin-hadoop1.tgz>`_ link to download the latest version (Spark 1.3) now. If you are interested in Spark development, you can instead download the sources and follow `these instructions <http://spark.apache.org/docs/latest/building-spark.html>`_ to build from source.

Once you have downloaded Spark, set an environmental variable by typing the following into the terminal (here we assume you downloaded a pre-built version and put it in your downloads folder)

.. code-block:: bash

	export SPARK_HOME=~/downloads/spark-1.3.0-bin-hadoop1

To make this setting permanent, add the above line to your bash profile (usually located in ``~/.bash_profile`` on Mac OS X), and open a new terminal so the change takes effect. Otherwise, you'll need to enter this line during each terminal session.

Installing Thunder
~~~~~~~~~~~~~~~~~~
Type the following line into the terminal

.. code-block:: bash
	
	pip install thunder-python

If it didn't work, try

.. code-block:: bash
	
	sudo pip install thunder-python

If the installation was successful you will have three new commands available: ``thunder``, ``thunder-submit``, and ``thunder-ec2``. You can check the location of these commands by typing

.. code-block:: bash
	
	which thunder

If you type ``thunder`` into the terminal it will start an interactive session in the Python shell.

If you want to upgrade to the latest version, we recommend first uninstalling using 

.. code-block:: bash
	
	pip uninstall thunder-python

And then install again to get the latest version.

Dependencies 
~~~~~~~~~~~~
Along with Spark, Thunder depends on these Python libraries (by installing using ``pip`` these will be added automatically).

`numpy <http://www.numpy.org/>`_, `scipy <http://www.scipy.org/>`_, `matplotlib <matplotlib.sourceforge.net>`_, `scikit-learn <http://scikit-learn.org/stable/>`_ 

We recommend using the `Anaconda distribution <https://store.continuum.io/cshop/anaconda/>`_, which includes these dependencies (and many other useful packages). Especially if you aren't already using Python for scientific computing, it's a great way to start. 

IPython
~~~~~~~
Spark and Thunder work very well alongside IPython and the IPython notebook. When calling `thunder`, adding the flag ``-i`` will launch using IPython, and the flag ``-n`` will launch in an IPython notebook.

You can also use environmental variables to make these options more permanent. If you set

.. code-block:: bash

	export IPYTHON=1

when you type ``thunder`` it will start in IPython. If you additionally set

.. code-block:: bash

	export IPYTHON_OPTS="notebook"

it will use the IPython notebook.


