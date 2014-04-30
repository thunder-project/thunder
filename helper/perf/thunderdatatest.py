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
            time = (dt.microseconds + (dt.seconds + dt.days * 24.0 * 3600.0) * 10.0**6.0) / 10.0**6.0
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


class KMeans(ThunderDataTest):

    def __init__(self, sc):
        ThunderDataTest.__init__(self, sc)

    def runtest(self):
        labels, centers = kmeans(self.rdd, 3, maxiter=5, tol=0)


class ICA(ThunderDataTest):

    def __init__(self, sc):
        ThunderDataTest.__init__(self, sc)

    def runtest(self):
        k = len(self.rdd.first()[1])
        c = 3
        n = 1000
        B = orth(random.randn(k, c))
        Bold = zeros((k, c))
        iterNum = 0
        errVec = zeros(20)
        while (iterNum < 5):
            iterNum += 1
            # update rule for pow3 nonlinearity (TODO: add other nonlins)
            B = self.rdd.map(lambda (_, v): v).map(lambda x: outer(x, dot(x, B) ** 3)).reduce(lambda x, y: x + y) / n - 3 * B
            # orthognalize
            B = dot(B, real(sqrtm(inv(dot(transpose(B), B)))))
            # evaluate error
            minAbsCos = min(abs(diag(dot(transpose(B), Bold))))
            # store results
            Bold = B
            errVec[iterNum-1] = (1 - minAbsCos)

        sigs = self.rdd.mapValues(lambda x: dot(B, x))


class PCA(ThunderDataTest):

    def __init__(self, sc):
        ThunderDataTest.__init__(self, sc)

    def runtest(self):
        m = len(self.rdd.first()[1])
        k = 3
        n = 1000

        def outerprod(x):
            return outer(x, x)

        c = random.rand(k, m)
        iter = 0
        error = 100

        while (iter < 5):
            c_old = c
            c_inv = dot(transpose(c), inv(dot(c, transpose(c))))
            premult1 = self.rdd.context.broadcast(c_inv)
            xx = self.rdd.map(lambda (_, v): v).map(lambda x: outerprod(dot(x, premult1.value))).sum()
            xx_inv = inv(xx)
            premult2 = self.rdd.context.broadcast(dot(c_inv, xx_inv))
            c = self.rdd.map(lambda (_, v): v).map(lambda x: outer(x, dot(x, premult2.value))).sum()
            c = transpose(c)
            error = sum(sum((c - c_old) ** 2))
            iter += 1


TESTS = {
    'stats': Stats,
    'average': Average,
    'regress': Regress,
    'regresswithsave': RegressWithSave,
    'crosscorr': CrossCorr,
    'fourier': Fourier,
    'load': Load,
    'save': Save,
    'ica': ICA,
    'pca': PCA,
    'kmeans': KMeans
}
