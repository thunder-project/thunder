
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

For a full list of methods, see...

## reading

Both `images` and `series` can be loaded from a variety of data types and locations. For all loading methods, the optional argument `engine` allows you to specify whether data should be loaded in `"local"` mode, which is backed by a `numpy` array, or in `"spark"` mode, which is backed by an RDD.

All loading methods are available on the module for the corresponding data type, for example

```python
import thunder as td
data = td.images.fromtif('/path/to/tifs')
data = td.series.fromarray(array)
```

### images

##### `fromarray(values, npartitions=None, engine=None)`

#### `frompng`

#### `fromtif`

#### `frombinary`

#### `fromexample`

#### `fromrandom`

##### `fromlist(items, accessor=None, keys=None, dims=None, dtype=None, npartitions=None, engine=None)`

#### `frompath`

#### `fromrdd`

### series

#### `fromrdd`

#### `fromlist`

#### `fromarray`

#### `frombinary`

#### `fromtext`

#### `frommat`

#### `fromnpy`

#### `fromrdd`

#### `fromrandom`

#### `fromexample`

## writing

### images

#### `images.tobinary()`

#### `images.topng()`

#### `images.totif()`

### series

#### `series.tobinary()`

## contributing

Thunder is a community effort! The codebase so far is due to the excellent work of the following individuals:

> Andrew Osheroff, Ben Poole, Chris Stock, Davis Bennett, Jascha Swisher, Jason Wittenbach, Jeremy Freeman, Josh Rosen, Kunal Lillaney, Logan Grosenick, Matt Conlen, Michael Broxton, Noah Young, Ognen Duzlevski, Richard Hofer, Owen Kahn, Ted Fujimoto, Tom Sainsbury, Uri Laseron

If you run into a problem, have a feature request, or want to contribute, submit an issue or a pull request, or come talk to us in the [chatroom](https://gitter.im/thunder-project/thunder).