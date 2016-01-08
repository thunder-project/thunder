import checkist
from numpy import sqrt, pi, angle, fft, fix, zeros, roll, dot, mean, \
    array, size, asarray, polyfit, polyval, arange, percentile, ceil, float64, where

from ..series.series import Series


class TimeSeries(Series):
    """
    Collection of time series data.
    """
    @property
    def _constructor(self):
        return TimeSeries

    def _makewindows(self, indices, window):
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

    def mean_by_window(self, indices, window):
        """
        Average time series across multiple windows specified by their centers

        Parameters
        ----------
        indices : array-like
            List of times specifying window centers

        window : int
            Window size
        """
        masks = self._makewindows(indices, window)
        newindex = arange(0, len(masks[0]))
        return self.map(lambda x: mean([x[m] for m in masks], axis=0), index=newindex)

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
        newindex = self.index[s]
        return self.map(lambda v: v[s], index=newindex)

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
            raise Exception('Requested frequency, %g, is too high, '
                            'must be less than half the series duration' % freq)

        index = ['coherence', 'phase']
        return self.map(lambda x: get(x, freq), index=index)

    def convolve(self, signal, mode='full', var=None):
        """
        Conolve time series data against another signal

        Parameters
        ----------
        signal : array
            Signal to convolve with (must be 1D)

        var : str
            Variable name if loading from a MAT file

        mode : str, optional, default='full'
            Mode of convolution, options are 'full', 'same', and 'same'
        """

        from numpy import convolve

        s = asarray(signal)

        n = size(self.index)
        m = size(s)

        # use expected lengths to make a new index
        if mode == 'same':
            newmax = max(n, m)
        elif mode == 'valid':
            newmax = max(m, n) - min(m, n) + 1
        else:
            newmax = n+m-1
        newindex = arange(0, newmax)

        return self.map(lambda x: convolve(x, signal, mode), index=newindex)

    def crosscorr(self, signal, lag=0):
        """
        Cross correlate time series data against another signal

        Parameters
        ----------
        signal : array
            Signal to correlate against (must be 1D)

        lag : int
            Range of lags to consider, will cover (-lag, +lag)
        """
        from scipy.linalg import norm

        s = asarray(signal)
        s = s - mean(s)
        s = s / norm(s)

        if size(s) != size(self.index):
            raise Exception('Size of signal to cross correlate with, %g, does not match size of series' % size(s))

        # created a matrix with lagged signals
        if lag is not 0:
            shifts = range(-lag, lag+1)
            d = len(s)
            m = len(shifts)
            sshifted = zeros((m, d))
            for i in range(0, len(shifts)):
                tmp = roll(s, shifts[i])
                if shifts[i] < 0:
                    tmp[(d+shifts[i]):] = 0
                if shifts[i] > 0:
                    tmp[:shifts[i]] = 0
                sshifted[i, :] = tmp
            s = sshifted
        else:
            shifts = [0]

        def get(y, s):
            y = y - mean(y)
            n = norm(y)
            if n == 0:
                b = zeros((s.shape[0],))
            else:
                y /= norm(y)
                b = dot(s, y)
            return b

        return self.map(lambda x: get(x, s), index=shifts)

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
        checkist.opts(method, ['linear', 'nonlinear'])

        if method == 'linear':
            order = 1

        if method == 'nonlinear':
            if 'order' in kwargs:
                order = kwargs['order']
            else:
                order = 5

        def func(y):
            x = arange(len(y))
            p = polyfit(x, y, order)
            p[-1] = 0
            yy = polyval(p, x)
            return y - yy

        return self.map(func)

    def normalize(self, method='percentile', window=None, perc=20, offset=0.1):
        """
        Normalize each time series by subtracting and dividing by a baseline.

        Baseline can be derived from a global mean or percentile,
        or a smoothed percentile estimated within a rolling window.

        Parameters
        ----------
        baseline : str, optional, default = 'percentile'
            Quantity to use as the baseline, options are 'mean', 'percentile',
            'window', or 'window-exact'.

        window : int, optional, default = 6
            Size of window for baseline estimation,
            for 'window' and 'window-exact' baseline only.

        perc : int, optional, default = 20
            Percentile value to use, for 'percentile',
            'window', or 'window-exact' baseline only.

        offset : float, optional, default = 0.1
             Scalar added to baseline during division to avoid division by 0.
        """
        checkist.opts(method, ['mean', 'percentile', 'window', 'window-exact'])
    
        from warnings import warn
        if not (method == 'window' or method == 'window-exact') and window is not None:
            warn('Setting window without using method "window" has no effect')

        if method == 'mean':
            baseFunc = mean

        if method == 'percentile':
            baseFunc = lambda x: percentile(x, perc)

        if method == 'window':
            from scipy.ndimage.filters import percentile_filter
            baseFunc = lambda x: percentile_filter(x.astype(float64), perc, window, mode='nearest')

        if method == 'window-exact':
            if window & 0x1:
                left, right = (ceil(window/2), ceil(window/2) + 1)
            else:
                left, right = (window/2, window/2)

            n = len(self.index)
            baseFunc = lambda x: asarray([percentile(x[max(ix-left, 0):min(ix+right+1, n)], perc)
                                          for ix in arange(0, n)])

        def get(y):
            b = baseFunc(y)
            return (y - b) / (b + offset)

        return self.map(get)