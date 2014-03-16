#!/usr/bin/env python

import glob
import os
egg = glob.glob(os.path.join(os.environ['THUNDER_EGG'], "*.egg"))
sc.addPyFile(egg[0])
from thunder.util.load import load, getdims
from thunder.util.save import save
from thunder.regression.regress import regress
from thunder.factorization.pca import pca
from thunder.sigprocessing.stats import stats
