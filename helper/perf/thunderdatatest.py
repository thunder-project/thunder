import abc
import os
from datetime import datetime
from numpy import arange, add, float16
from scipy.io import savemat
from thunder.util.load import load
from thunder.sigprocessing.util import SigProcessingMethod
from thunder.regression.util import RegressionModel

class ThunderDataTest(object):

    def __init__(self, sc):
        self.sc = sc

    @abc.abstractmethod
    def runtest(self, **args):
        return

    @staticmethod
    def initialize(testname, sc):
        return TESTS[testname](sc)

    def createinputdata(self, numrecords, numdims, numpartitions):
        rdd = self.sc.parallelize(arange(0, numrecords), numpartitions)
        self.rdd = rdd

    def loadinputdata(self, datafile, savefile=None):
        rdd = load(self.sc, datafile, preprocessmethod="dff-percentile")
        self.rdd = rdd
        self.datafile = datafile
        if savefile is not None:
            self.savefile = savefile
        self.modelfile = os.path.join(os.path.split(self.datafile)[0], 'stim')

    def run(self, numtrials, persistencetype):

        if persistencetype == "memory":
            self.rdd.cache()
            self.rdd.count()

        def timedtest(func):
            start = datetime.now()
            func()
            end = datetime.now()
            dt = end - start
            time = dt.total_seconds()
            return time

        results = map(lambda i: timedtest(self.runtest), range(0, numtrials))

        return results


class Stats(ThunderDataTest):

    def __init__(self, sc):
        ThunderDataTest.__init__(self, sc)
        self.method = SigProcessingMethod.load("stats", statistic="std")

    def runtest(self):
        vals = self.method.calc(self.rdd)
        vals.count()


class Average(ThunderDataTest):

    def __init__(self, sc):
        ThunderDataTest.__init__(self, sc)

    def runtest(self):
        vec = self.rdd.map(lambda (_, v): v).mean()


class Regress(ThunderDataTest):

    def __init__(self, sc):
        ThunderDataTest.__init__(self, sc)

    def runtest(self):
        model = RegressionModel.load(os.path.join(self.modelfile, "linear"), "linear")
        betas, stats, resid = model.fit(self.rdd)
        stats.count()


class RegressWithSave(ThunderDataTest):

    def __init__(self, sc):
        ThunderDataTest.__init__(self, sc)

    def runtest(self):
        model = RegressionModel.load(os.path.join(self.modelfile, "linear"), "linear")
        betas, stats, resid = model.fit(self.rdd)
        result = stats.map(lambda (_, v): float16(v)).collect()
        savemat(self.savefile + "tmp.mat", mdict={"tmp": result}, oned_as='column')


class CrossCorr(ThunderDataTest):

    def __init__(self, sc):
        ThunderDataTest.__init__(self, sc)

    def runtest(self):
        method = SigProcessingMethod.load("crosscorr", sigfile=os.path.join(self.modelfile, "crosscorr"), lag=0)
        betas = method.calc(self.rdd)
        betas.count()


class Fourier(ThunderDataTest):

    def __init__(self, sc):
        ThunderDataTest.__init__(self, sc)
        self.method = SigProcessingMethod.load("fourier", freq=5)

    def runtest(self):
        vals = self.method.calc(self.rdd)
        vals.count()


class Load(ThunderDataTest):

    def __init__(self, sc):
        ThunderDataTest.__init__(self, sc)

    def runtest(self):
        self.rdd.count()


class Save(ThunderDataTest):

    def __init__(self, sc):
        ThunderDataTest.__init__(self, sc)

    def runtest(self):
        result = self.rdd.map(lambda (_, v): float16(v[0])).collect()
        savemat(self.savefile + "tmp.mat", mdict={"tmp": result}, oned_as='column')


TESTS = {
    'stats': Stats,
    'average': Average,
    'regress': Regress,
    'regresswithsave': RegressWithSave,
    'crosscorr': CrossCorr,
    'fourier': Fourier,
    'load': Load,
    'save': Save
}
