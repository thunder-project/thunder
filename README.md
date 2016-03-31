
# thunder

[![Latest Version](https://img.shields.io/pypi/v/thunder-python.svg?style=flat-square)](https://pypi.python.org/pypi/thunder-python)
[![Build Status](https://img.shields.io/travis/thunder-project/thunder/master.svg?style=flat-square)](https://travis-ci.org/thunder-project/thunder) 
[![Gitter](https://img.shields.io/gitter/room/thunder-project/thunder.svg?style=flat-square)](https://gitter.im/thunder-project/thunder)

> large-scale image and time series analysis in python

Thunder is an ecosystem of tools for the analysis of image and time series data in Python. It provides data structures and algorithms for loading, processing, and analyzing these  data, and can be useful in a variety of domains, including neuroscience, medical imaging, video processing, and geospatial and climate analysis. It can be used locally, but also supports large-scale analysis through the distributed computing engine [Spark](https://github.com/apache/spark).

## install

The core `thunder` package defines data structures and read/write patterns for images and time series. It is built on `numpy`, `scipy`, `scikit-learn`, and `scikit-image`, and is compatible with Python 2.7 and 3.4. You can install it using:

```
pip install thunder-python
```

If you want to install all related packages at the same time, you can use

```
pip install thunder-python[all]
```

This will additionally install `thunder-regression`, `thunder-registration`, and `thunder-factoriation`.

## example

## usage

## data types

Primary data types are `images` and `series`. 

Basic description.

For a full list of methods, see [link]() for complete API documentation.

## reading

Both `images` and `series` can be loaded from a variety of data types and locations. For all loading methods, the optional argument `engine` allows you to specify whether data should be loaded in `"local"` mode, which is backed by a `numpy` array, or in `"spark"` mode, which is backed by an RDD.

All loading methods are available on the module for the corresponding data type, for example

```python
import thunder as td
data = td.images.fromtif('/path/to/tifs')
data = td.series.fromarray(array)
```

Every loading method has the optional argument `engine` which can be either `None` (for local use) or a `SparkContext` (for distributed use with Spark). And all methods that load from files e.g. `fromtif` or `frombinary` can load from either a local filesystem or Amazon S3, with the optional argument `credentials` for loading non-public data from S3. An overview of available methods is below; see [link]() for complete API documentation.

### images

#### `frompng(path, opts)`

#### `fromtif(path, opts)`

#### `frombinary(path, opts)`

#### `fromarray(values, opts)`

#### `fromlist(items, opts)`

#### `fromrdd(rdd, opts)`

#### `fromrandom(shape, opts)`

#### `fromexample(name, opts)`

### series

#### `frombinary(path, opts)`

#### `fromtext(path, opts)`

#### `fromarray(values, opts)`

#### `fromlist(items, opts)`

#### `fromrdd(rdd, opts)`

#### `fromrandom(shape, opts)`

#### `fromexample(name, opts)`

## writing

All `Images` and `Series` objects can be written to disk in a few different formats. As with reading, data can be written to either a local filesystem or to Amazon S3, with credentials provided as an optional argument. The methods are summarized below, see [link]() for the complete API documentation.

### images

#### `images.tobinary(opts)`

#### `images.topng(opts)`

#### `images.totif(opts)`

### series

#### `series.tobinary(opts)`

## contributing

Thunder is a community effort! The codebase so far is due to the excellent work of the following individuals:

> Andrew Osheroff, Ben Poole, Chris Stock, Davis Bennett, Jascha Swisher, Jason Wittenbach, Jeremy Freeman, Josh Rosen, Kunal Lillaney, Logan Grosenick, Matt Conlen, Michael Broxton, Noah Young, Ognen Duzlevski, Richard Hofer, Owen Kahn, Ted Fujimoto, Tom Sainsbury, Uri Laseron

If you run into a problem, have a feature request, or want to contribute, submit an issue or a pull request, or come talk to us in the [chatroom](https://gitter.im/thunder-project/thunder).