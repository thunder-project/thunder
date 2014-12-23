from numpy import sqrt, pi, angle, fft, fix, zeros, roll, dot, mean, \
    array, size, diag, tile, ones, asarray, polyfit, polyval, arange, \
    percentile, ceil

from thunder.rdds.series import Series
from thunder.utils.common import loadmatvar, checkparams


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

    def triggeredAverage(self, events, lag=0):
        """
        Construct an average time series triggered on each of several events,
        considering a range of lags before and after the event

        Parameters
        ----------
        events : array-like
            List of events to trigger on

        lag : int
            Range of lags to consider, will cover (-lag, +lag)
        """
        events = asarray(events)
        m = zeros((lag*2+1, len(self.index)))
        for i, shift in enumerate(range(-lag, lag+1)):
            fillinds = events + shift
            fillinds = fillinds[fillinds >= 0]
            fillinds = fillinds[fillinds < len(self.index)]
            m[i, fillinds] = 1

        if lag == 0:
            newindex = 0
        else:
            newindex = range(-lag, lag+1)

        scale = m.sum(axis=1)

        rdd = self.rdd.mapValues(lambda x: dot(m, x) / scale)
        return self._constructor(rdd, index=newindex).__finalize__(self)

    def blockedAverage(self, blocklength):
        """
        Average blocks of a time series together, e.g. because they correspond
        to trials of some repeated measurement or process

        Parameters
        ----------
        triallength : int
            Length of trial, must divide evenly into total length of time series
        """

        n = len(self.index)

        if divmod(n, blocklength)[1] != 0:
            raise Exception('Trial length, %g, must evenly divide length of time series, %g'
                            % (blocklength, n))

        if n == blocklength:
            raise Exception('Trial length, %g, cannot be length of entire time series, %g'
                            % (blocklength, n))

        m = tile(diag(ones((blocklength,))), [n/blocklength, 1]).T
        newindex = range(0, blocklength)
        scale = n / blocklength

        rdd = self.rdd.mapValues(lambda x: dot(m, x) / scale)
        return self._constructor(rdd, index=newindex).__finalize__(self)

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
            amp_ft = 2*abs(ft)/nframes
            amp = amp_ft[freq]
            amp_sum = sqrt(sum(amp_ft**2))
            co = amp / amp_sum
            ph = -(pi/2) - angle(ft[freq])
            if ph < 0:
                ph += pi * 2
            return array([co, ph])

        if freq >= int(fix(size(self.index)/2)):
            raise Exception('Requested frequency, %g, is too high, must be less than half the series duration' % freq)

        rdd = self.rdd.mapValues(lambda x: get(x, freq))
        return Series(rdd, index=['coherence', 'phase']).__finalize__(self)

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
            s = loadmatvar(signal, var)
        else:
            s = signal

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
            s_shifted = zeros((m, d))
            for i in range(0, len(shifts)):
                tmp = roll(s, shifts[i])
                if shifts[i] < 0:  # zero padding
                    tmp[(d+shifts[i]):] = 0
                if shifts[i] > 0:
                    tmp[:shifts[i]] = 0
                s_shifted[i, :] = tmp
            s = s_shifted
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
        checkparams(method, ['linear', 'nonlin'])

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

        return self.applyValues(func)

    def normalize(self, baseline='percentile', window=6, perc=20):
        """ Normalize each time series by subtracting and dividing by a baseline.

        Baseline can be derived from a global mean or percentile,
        or a smoothed percentile estimated within a rolling window.

        Parameters
        ----------
        baseline : str, optional, default = 'percentile'
            Quantity to use as the baseline, options are 'mean', 'percentile', or 'window'

        perc : int, optional, default = 20
            Percentile value to use, for 'percentile' or 'window' baseline only

        window : int, optional, default = 6
            Size of window for baseline estimation, for 'window' baseline only
        """
        checkparams(baseline, ['mean', 'percentile', 'window'])
        method = baseline.lower()

        if method == 'mean':
            basefunc = mean

        if method == 'percentile':
            basefunc = lambda x: percentile(x, perc)

        # TODO optimize implementation by doing a single initial sort
        if method == 'window':
            if window & 0x1:
                left, right = (ceil(window/2), ceil(window/2) + 1)
            else:
                left, right = (window/2, window/2)

            n = len(self.index)
            basefunc = lambda x: asarray([percentile(x[max(ix-left, 0):min(ix+right+1, n)], perc)
                                          for ix in arange(0, n)])

        def get(y):
            b = basefunc(y)
            return (y - b) / (b + 0.1)

        return self.applyValues(get)

