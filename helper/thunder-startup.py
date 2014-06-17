#!/usr/bin/env python

import glob
import os

egg = glob.glob(os.path.join(os.environ['THUNDER_EGG'], "*.egg"))
if len(egg) > 0:
	sc.addPyFile(egg[0])
	print('Successfully added Thunder egg')
else:
	print('Warning: Thunder egg not found. If you are running on a cluster this file needs to be created. Make sure you have built an egg and that the env variable THUNDER_EGG is set')

from thunder.io import load, save, getdims
from thunder.clustering import KMeans
from thunder.regression import RegressionModel, TuningModel
from thunder.factorization import PCA
from thunder.factorization import ICA
from thunder.factorization import SVD
from thunder.timeseries import LocalCorr, Query, Fourier, Stats, CrossCorr
