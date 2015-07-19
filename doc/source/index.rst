.. meta::
   :description: Official documentation for the Thunder library
   :keywords: spark, big data, neuroscience, time series, image processing

.. title::
  thunder

.. raw:: html

  <img src='_static/header-logo.svg' width=700></img>

Spatial and temporal data is all around us, whether images from satellites or time series from electronic or biological sensors. These kinds of data are also the bread and butter of neuroscience. Almost all raw neural data consists of electrophysiological time series, or time-varying images of flourescence or resonance.

Thunder is a library for large-scale image and time series processing. It's designed for both local and distributed settings, and enables fast computation through the cluster computing platform Spark, bringing a crucial domain-specific applciation to this general-purpose computing engine.

It includes utitilies for loading and saving different image and binary formats, classes for working with distributed spatial and temporal data, and modular functions for analysis, factorization, and model fitting. It is written entirely in Python, and makes use of ``numpy``, ``scipy``, and ``scikit-learn`` on top of Spark's Python API, PySpark. It is designed for interactive use in notebooks (e.g. Juptyr) or through parameterized jobs.

This documentation is divided into into a general introduction_, instructions for installing Thunder either locally_ or on EC2_ through Amazon Web Services, an API explanation_ and reference_, a set of tutorials_ introducting key concepts, and information for developers_.

For more information, visit the GitHub_ repo, the `project page`_, or join our chatroom_.

.. _introduction: overview.html
.. _locally: local.html
.. _EC2: ec2.html
.. _explanation:
.. _reference:
.. _tutorials:
.. _developers:
.. _GitHub: https://github.com/thunder-project/thunder
.. _project page: https://thunder-project.org
.. _chatroom: https://gitter.im/thunder-project/thunder

.. toctree::
   :maxdepth: 1
   :hidden:

   installation
   deployment
   api
   contributing
   tutorials
