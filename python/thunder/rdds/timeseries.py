from numpy import sqrt, pi, angle, fft, fix, zeros, roll, dot, mean, \
    array, size, diag, tile, ones, asarray

from thunder.rdds.series import Series
from thunder.utils.common import loadMatVar


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
            fillInds = events + shift
            fillInds = fillInds[fillInds >= 0]
            fillInds = fillInds[fillInds < len(self.index)]
            m[i, fillInds] = 1

        if lag == 0:
            newIndex = 0
        else:
            newIndex = range(-lag, lag+1)

        scale = m.sum(axis=1)

        rdd = self.rdd.mapValues(lambda x: dot(m, x) / scale)
        return self._constructor(rdd, index=newIndex).__finalize__(self)

    def blockedAverage(self, blockLength):
        """
        Average blocks of a time series together, e.g. because they correspond
        to trials of some repeated measurement or process

        Parameters
        ----------
        triallength : int
            Length of trial, must divide evenly into total length of time series
        """

        n = len(self.index)

        if divmod(n, blockLength)[1] != 0:
            raise Exception('Trial length, %g, must evenly divide length of time series, %g'
                            % (blockLength, n))

        if n == blockLength:
            raise Exception('Trial length, %g, cannot be length of entire time series, %g'
                            % (blockLength, n))

        m = tile(diag(ones((blockLength,))), [n/blockLength, 1]).T
        newIndex = range(0, blockLength)
        scale = n / blockLength

        rdd = self.rdd.mapValues(lambda x: dot(m, x) / scale)
        return self._constructor(rdd, index=newIndex).__finalize__(self)

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
