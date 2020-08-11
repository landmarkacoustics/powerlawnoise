# Copyright (C) 2020 by Landmark Acoustics LLC
r'''Find the slope of a power spectrum.'''

import numpy as np


PLN_LOG_2 = np.log10(2.0)


class SpectralSlopeFinder:
    r'''Computes the slope of the power spectrum of a time series.

    Parameters
    ----------
    fft_size : int, optional
        The initial expected lengths of the time series inputs.

    Attributes
    ----------
    frequencies : np.ndarray
        The log10-valued frequencies for the x-axis of the slope.

    Examples
    --------
    >>> s = SpectralSlopeFinder()
    >>> rg = np.random.default_rng(42)
    >>> x = rg.standard_normal(2048)
    >>> y = s.spectrum(x)
    >>> def rough_rm(x, accuracy=15):
    ...     return np.around(np.sqrt(np.mean(x)), accuracy)
    ...
    >>> rough_rm(np.square(x)) == rough_rm(np.power(10.0, y))
    True
    >>> s(x)
    -0.058453950467672884
    >>> s(y)
    -0.040041342822683304

    '''

    def __init__(self, fft_size: int = 0):
        self._make_frequencies(max(0, fft_size))

    def _make_frequencies(self, fft_size):
        self._frequencies = np.r_[1.0/fft_size**2,
                                  np.linspace(1/fft_size, 0.5, fft_size//2)]
        self._frequencies = np.log10(self._frequencies)

    @property
    def frequencies(self) -> np.ndarray:
        return self._frequencies

    @staticmethod
    def spectrum(x: np.ndarray) -> np.ndarray:
        r'''Computes the log10-valued power spectrum of a vector

        Parameters
        ----------
        x : np.ndarray
            The input data. Its length should be at least 2.

        Returns
        -------
        np.ndarray : a log10-valued power spectrum for frequencies [0, Nyquist]

        Notes
        -----
        The spectrum uses numpy.fft.rfft but rescales the value so that the
        spectrum's RMS is equal to the RMS of the input array.

        See Also
        --------
        numpy.fft.rfft

        '''

        fft_size = len(x)
        if fft_size < 2:
            return np.repeat(np.nan, 1 + fft_size//2)

        y = 2*np.log10(np.abs(np.fft.rfft(x))) - np.log10(fft_size)
        y[0] -= PLN_LOG_2
        y[-1] -= PLN_LOG_2
        y += np.log10(len(y)) - np.log10(len(y) - 1)
        return y

    def __call__(self, noise: np.ndarray) -> float:
        r'''Find the log-log slope of the power spectrum of `noise`.

        Parameters
        ----------
        noise : np.ndarray
            The time series to examine. It should be at least 6 samples long.

        Returns
        -------
        float : the slope of the spectrum for frequencies in (0, Nyquist).

        '''

        if len(noise) < 6:
            return np.nan

        spectrum = self.spectrum(noise)
        if len(spectrum) != len(self._frequencies):
            self._make_frequencies(len(noise))
        covariance = np.cov(self._frequencies[1:], spectrum[1:])
        return covariance[0, 1] / covariance[0, 0]
