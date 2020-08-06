# Copyright (C) 2020 by Landmark Acoustics LLC
r'''A class that convolves white noise into power law noise.
'''


import numpy as np

from .coefficients import \
    coefficient_array


class PowerLawNoise:
    r'''All the precomputations needed to generate power-law noise.

    Parameters
    ----------
    alpha : float
        The exponent that describes the shape of the noise (red/Brownian = -2).
    degree : int
        The number of terms in the autoregressive model.

    Attributes
    ----------
    alpha : float
        The exponent of the power law.
    AR_terms : np.ndarray
        The terms of the AR models from h[degree] to h[1].
    degree : int
        The number of terms in the autoregressive (AR) model.
    terms : np.ndarray
        The terms of the AR model from h[0] to h[degree].

    See Also
    --------
    power_law_noise.coefficient_array : makes the AR terms

    Examples
    --------
    >>> x = np.linspace(0.25, 1, 4, dtype=float)
    >>> red = PowerLawNoise(-2, 4)
    >>> red.terms
    array([ 1., -1., -0., -0., -0.])
    >>> red(x)
    1.0
    >>> pink = PowerLawNoise(-1, 4)
    >>> pink.terms
    array([ 1.       , -0.5      , -0.125    , -0.0625   , -0.0390625])
    >>> pink(x)
    0.634765625
    >>> pink(x[1:])
    0.625
    >>> pink(x[:-1])
    0.453125
    >>> white = PowerLawNoise(0, 4)
    >>> white.terms
    array([1., 0., 0., 0., 0.])
    >>> white(x)
    0.0
    >>> blue = PowerLawNoise(1, 4)
    >>> blue.terms
    array([1.       , 0.5      , 0.375    , 0.3125   , 0.2734375])
    >>> blue(x)
    -1.005859375
    >>> violet = PowerLawNoise(2, 4)
    >>> violet.terms
    array([1., 1., 1., 1., 1.])
    >>> violet(x)
    -2.5
    '''

    def __init__(self, alpha: float, degree: int):
        self._a = alpha
        self._d = degree
        self._h = coefficient_array(self._a, self._d)[-1::-1]
        self._buffer = np.zeros(0, dtype=float)

    @property
    def alpha(self) -> float:
        return self._a

    @property
    def degree(self) -> int:
        return self._d

    @property
    def terms(self) -> np.ndarray:
        return self._h

    def transfer_function(self, fft_size: int) -> np.ndarray:
        '''A complex-valued set of terms that describe the frequency response.

        Parameters
        ----------
        fft_size : int
            The number of frames in the transfer function.

        Returns
        -------
        transfer_function : np.ndarray
            Complex-valued of length fft_size

        See Also
        --------
        numpy.fft.fft : computes the Discreet Fourier Transform

        '''

        n_terms = self._d + 1
        if fft_size < n_terms:
            raise ValueError(f'The fft_size must be at least {n_terms}.')

        return np.fft.fft(np.r_[self.terms, np.zeros(fft_size - n_terms)])

    def __call__(self, noise: np.ndarray) -> float:
        r'''Returns either the predicted value or the AR effect.

        Parameters
        ----------
        noise : np.ndarray
            `len(noise)` must be at most `self.degree`

        Returns
        -------
        power_law_noise : float
            The dot product of `noise` and either [`AR_terms`, 1] or `AR_terms`

        '''

        return np.dot(noise, self._h[-len(noise):])

    def generate_noise(self,
                       input_source: np.ndarray,
                       degree: int = None) -> float:
        r'''Generate power law noise by iterating through `input_source`.

        Parameters
        ----------
        input_source: np.ndarray
            The noise or impulse or whatever that drives the power law output.
        degree : int, optional
            The degree of the model. Must be lte the default, `self.degree`.

        Yields
        ------
        sample : float
            One instant of power law noise.

        '''

        if degree is None:
            degree = self.degree

        if degree > 0:
            buf = np.zeros(degree)
            for x in input_source:
                np.roll(buf, -1)
                buf[-1] = x + np.dot(self._h[:-1][-degree:], buf)
                yield buf[-1]
        else:
            for x in input_source:
                yield x

    def __repr__(self) -> str:
        return f'Law(alpha={self.alpha}, degree=={self.degree})'
