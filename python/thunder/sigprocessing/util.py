"""
utilities for signal processing
"""

from numpy import sqrt, fix, pi, median, std, sum, mean, shape, zeros, roll, dot, angle, abs
from scipy.linalg import norm
from scipy.io import loadmat
from numpy.fft import fft


class SigProcessingMethod(object):
    """class for doing signal processing"""

    @staticmethod
    def load(method, **opts):
        return SIGPROCESSING_METHODS[method](**opts)

    def get(self, y):
        pass

    def calc(self, data):
        result = data.map(lambda x: self.get(x))
        return result


class FourierMethod(SigProcessingMethod):
    """class for computing fourier transform"""

    def __init__(self, freq):
        """get frequency"""

        self.freq = freq

    def get(self, y):
        """compute fourier amplitude (coherence) and phase"""

        y = y - mean(y)
        nframes = len(y)
        ft = fft(y)
        ft = ft[0:int(fix(nframes/2))]
        amp_ft = 2*abs(ft)/nframes
        amp = amp_ft[self.freq]
        amp_sum = sqrt(sum(amp_ft**2))
        co = amp / amp_sum
        ph = -(pi/2) - angle(ft[self.freq])
        if ph < 0:
            ph += pi * 2
        return co, ph


class StatsMethod(SigProcessingMethod):
    """class for computing simple summary statistics"""

    def __init__(self, statistic):
        """get mode"""
        self.func = {
            'median': lambda x: median(x),
            'mean': lambda x: mean(x),
            'std': lambda x: std(x),
            'norm': lambda x: norm(x - mean(x)),
        }[statistic]

    def get(self, y):
        """compute fourier amplitude (coherence) and phase"""

        return self.func(y)


class QueryMethod(SigProcessingMethod):
    """class for computing averages over indices"""

    def __init__(self, indsfile):
        """get indices"""
        if type(indsfile) is str:
            inds = loadmat(indsfile)['inds'][0]
        else:
            inds = indsfile
        self.inds = inds
        self.n = len(inds)


class CrossCorrMethod(SigProcessingMethod):
    """class for computing lagged cross correlations"""

    def __init__(self, sigfile, lag):
        """load parameters. paramfile can be an array, or a string
        if its a string, assumes signal is a MAT file
        with name modelfile_X
        """
        if type(sigfile) is str:
            x = loadmat(sigfile + "_X.mat")['X'][0]
        else:
            x = sigfile
        x = x - mean(x)
        x = x / norm(x)

        if lag is not 0:
            shifts = range(-lag, lag+1)
            d = len(x)
            m = len(shifts)
            x_shifted = zeros((m, d))
            for ix in range(0, len(shifts)):
                tmp = roll(x, shifts[ix])
                if shifts[ix] < 0:  # zero padding
                    tmp[(d+shifts[ix]):] = 0
                if shifts[ix] > 0:
                    tmp[:shifts[ix]] = 0
                x_shifted[ix, :] = tmp
            self.x = x_shifted
        else:
            self.x = x

    def get(self, y):
        """compute cross correlation between y and x"""

        y = y - mean(y)
        n = norm(y)
        if n == 0:
            b = zeros((shape(self.x)[0],))
        else:
            y /= norm(y)
            b = dot(self.x, y)
        return b


SIGPROCESSING_METHODS = {
    'stats': StatsMethod,
    'fourier': FourierMethod,
    'crosscorr': CrossCorrMethod,
    'query': QueryMethod
}