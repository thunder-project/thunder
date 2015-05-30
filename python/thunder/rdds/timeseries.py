from numpy import sqrt, pi, angle, fft, fix, zeros, roll, dot, mean, \
    array, size, asarray, polyfit, polyval, arange, \
    percentile, ceil, float64, where, floor

from thunder.rdds.series import Series
from thunder.utils.common import loadMatVar, checkParams


class TimeSeries(Series):
    """
    Distributed collection of time series data.

    Backed by an RDD of key-value pairs where the key is an identifier
    and the value is a one-dimensional array. The common index
    specifies the time of each entry in the array.

    Parameters
    ----------
    rdd : RDD of (tuple, array) pairs
        RDD containing the series data.

    index : array-like
        Time indices, must be same length as the arrays in the input data.
        Defaults to arange(len(data)) if not provided.

    dims : Dimensions
        Specify the dimensions of the keys (min, max, and count), can
        avoid computation if known in advance.

    See also
    --------
    Series : base class for Series data
    """
    # use superclass __init__

    @property
    def _constructor(self):
        return TimeSeries

    def _makeWindowMasks(self, indices, window):
        """
        Make masks used by windowing functions

        Given a list of indices specifying window centers,
        and a window size, construct a list of index arrays,
        one per window, that index into the target array

        Parameters
        ----------
        indices : array-like
            List of times specifying window centers

        window : int
            Window size
        """
        before = window / 2
        after = window / 2 + divmod(window, 2)[1]
        index = asarray(self.index)
        indices = asarray(indices)
        if where(index == max(indices))[0][0] + after > len(index):
            raise ValueError("Maximum requested index %g, with window %g, exceeds length %g"
                             % (max(indices), window, len(index)))
        if where(index == min(indices))[0][0] - before < 0:
            raise ValueError("Minimum requested index %g, with window %g, is less than 0"
                             % (min(indices), window))
        masks = [arange(where(index == i)[0][0]-before, where(index == i)[0][0]+after) for i in indices]
        return masks

    def meanByWindow(self, indices, window):
        """
        Average time series across multiple windows specified by their centers

        Parameters
        ----------
        indices : array-like
            List of times specifying window centers

        window : int
            Window size
        """
        masks = self._makeWindowMasks(indices, window)
        rdd = self.rdd.mapValues(lambda x: mean([x[m] for m in masks], axis=0))
        index = arange(0, len(masks[0]))
        return self._constructor(rdd, index=index).__finalize__(self)

    def groupByWindow(self, indices, window):
        """
        Group time series into multiple windows specified by their centers

        Parameters
        ----------
        indices : array-like
            List of times specifying window centers

        window : int
            Window size
        """
        masks = self._makeWindowMasks(indices, window)
        rdd = self.rdd.flatMap(lambda (k, x): [(k + (i, ), x[m]) for i, m in enumerate(masks)])
        index = arange(0, len(masks[0]))
        nrecords = self.nrecords * len(indices)
        return self._constructor(rdd, index=index, nrecords=nrecords).__finalize__(self)

    def subsample(self, sampleFactor=2):
        """
        Subsample time series by an integer factor

        Parameters
        ----------
        sampleFactor : positive integer, optional, default=2

        """
        if sampleFactor < 0:
            raise Exception('Factor for subsampling must be postive, got %g' % sampleFactor)
        s = slice(0, len(self.index), sampleFactor)
        newIndex = self.index[s]
        return self._constructor(
            self.rdd.mapValues(lambda v: v[s]), index=newIndex).__finalize__(self)

    def fourier(self, freq=None):
        """
        Compute statistics of a Fourier decomposition on time series data

        Parameters
        ----------
        freq : int
            Digital frequency at which to compute coherence and phase
        """
        def get(y, freq):
            y = y - mean(y)
            nframes = len(y)
            ft = fft.fft(y)
            ft = ft[0:int(fix(nframes/2))]
            ampFt = 2*abs(ft)/nframes
            amp = ampFt[freq]
            ampSum = sqrt(sum(ampFt**2))
            co = amp / ampSum
            ph = -(pi/2) - angle(ft[freq])
            if ph < 0:
                ph += pi * 2
            return array([co, ph])

        if freq >= int(fix(size(self.index)/2)):
            raise Exception('Requested frequency, %g, is too high, must be less than half the series duration' % freq)

        rdd = self.rdd.mapValues(lambda x: get(x, freq))
        return Series(rdd, index=['coherence', 'phase']).__finalize__(self)

    def convolve(self, signal, mode='full', var=None):
        """
        Conolve time series data against another signal

        Parameters
        ----------
        signal : array, or str
            Signal to convolve with, can be a numpy array or a
            MAT file containing the signal as a variable

        var : str
            Variable name if loading from a MAT file

        mode : str, optional, default='full'
            Mode of convolution, options are 'full', 'same', and 'same'
        """

        from numpy import convolve

        if type(signal) is str:
            s = loadMatVar(signal, var)
        else:
            s = asarray(signal)

        n = size(self.index)
        m = size(s)

        newrdd = self.rdd.mapValues(lambda x: convolve(x, signal, mode))

        # use expected lengths to make a new index
        if mode == 'same':
            newmax = max(n, m)
        elif mode == 'valid':
            newmax = max(m, n) - min(m, n) + 1
        else:
            newmax = n+m-1
        newindex = arange(0, newmax)

        return self._constructor(newrdd, index=newindex).__finalize__(self)

    def crossCorr(self, signal, lag=0, var=None):
        """
        Cross correlate time series data against another signal

        Parameters
        ----------
        signal : array, or str
            Signal to correlate against, can be a numpy array or a
            MAT file containing the signal as a variable

        var : str
            Variable name if loading from a MAT file

        lag : int
            Range of lags to consider, will cover (-lag, +lag)
        """
        from scipy.linalg import norm

        if type(signal) is str:
            s = loadMatVar(signal, var)
        else:
            s = asarray(signal)

        # standardize signal
        s = s - mean(s)
        s = s / norm(s)

        if size(s) != size(self.index):
            raise Exception('Size of signal to cross correlate with, %g, does not match size of series' % size(s))

        # created a matrix with lagged signals
        if lag is not 0:
            shifts = range(-lag, lag+1)
            d = len(s)
            m = len(shifts)
            sShifted = zeros((m, d))
            for i in range(0, len(shifts)):
                tmp = roll(s, shifts[i])
                if shifts[i] < 0:  # zero padding
                    tmp[(d+shifts[i]):] = 0
                if shifts[i] > 0:
                    tmp[:shifts[i]] = 0
                sShifted[i, :] = tmp
            s = sShifted
        else:
            shifts = 0

        def get(y, s):
            y = y - mean(y)
            n = norm(y)
            if n == 0:
                b = zeros((s.shape[0],))
            else:
                y /= norm(y)
                b = dot(s, y)
            return b

        rdd = self.rdd.mapValues(lambda x: get(x, s))
        return self._constructor(rdd, index=shifts).__finalize__(self)

    def detrend(self, method='linear', **kwargs):
        """
        Detrend time series data with linear or nonlinear detrending
        Preserve intercept so that subsequent steps can adjust the baseline

        Parameters
        ----------
        method : str, optional, default = 'linear'
            Detrending method

        order : int, optional, default = 5
            Order of polynomial, for non-linear detrending only
        """
        checkParams(method, ['linear', 'nonlinear'])

        if method.lower() == 'linear':
            order = 1
        else:
            if 'order' in kwargs:
                order = kwargs['order']
            else:
                order = 5

        def func(y):
            x = arange(1, len(y)+1)
            p = polyfit(x, y, order)
            p[-1] = 0
            yy = polyval(p, x)
            return y - yy

        return self.applyValues(func, keepIndex=True)

    def normalize(self, baseline='percentile', window=None, perc=20):
        """
        Normalize each time series by subtracting and dividing by a baseline.

        Baseline can be derived from a global mean or percentile,
        or a smoothed percentile estimated within a rolling window.

        Parameters
        ----------
        baseline : str, optional, default = 'percentile'
            Quantity to use as the baseline, options are 'mean', 'percentile', 'window', or 'window-fast'

        window : int, optional, default = 6
            Size of window for baseline estimation, for 'window' and 'window-fast' baseline only

        perc : int, optional, default = 20
            Percentile value to use, for 'percentile', 'window', or 'window-fast' baseline only
        """
        checkParams(baseline, ['mean', 'percentile', 'window', 'window-fast'])
        method = baseline.lower()
    
        from warnings import warn
        if not (method == 'window' or method == 'window-fast') and window is not None:
            warn('Setting window without using method "window" has no effect')

        if method == 'mean':
            baseFunc = mean

        if method == 'percentile':
            baseFunc = lambda x: percentile(x, perc)

        if method == 'window':
            if window & 0x1:
                left, right = (ceil(window/2), ceil(window/2) + 1)
            else:
                left, right = (window/2, window/2)

            n = len(self.index)
            baseFunc = lambda x: asarray([percentile(x[max(ix-left, 0):min(ix+right+1, n)], perc)
                                          for ix in arange(0, n)])

        if method == 'window-fast':
            from scipy.ndimage.filters import percentile_filter
            baseFunc = lambda x: percentile_filter(x.astype(float64), perc, window, mode='nearest')

        def get(y):
            b = baseFunc(y)
            return (y - b) / (b + 0.1)

        return self.applyValues(get, keepIndex=True)

