# Copyright (C) 2020 by Landmark Acoustics LLC
r'''An autoregressive model for generating power-law noise.

The method here is outlined in Kasdin, N. J. and T. Walter. 1992. "Discrete
simulation of power law noise." Proc. 1992 IEEE Frequency Control Symposium.
Hershey, PA. pp 274-283.

'''


from typing import Tuple

import numpy as np


def generate_ar_coefficients(alpha: float) -> float:
    r'''Coefficients for a discrete autoregressive model of power law noise.

    Parameters
    ----------
    alpha : float
        The exponent of the power law. Its spectrum will have this slope on a
        log-log plot.

    Yields
    ------
    h : float
        The next coefficient in the model. Starts with h_1 because h_0 is one.

    '''

    h = 1.0
    k = 0
    g = k - 1 + 0.5*alpha

    while True:
        k += 1
        g += 1.0
        h *= g / k
        yield h


def coefficient_array(alpha: float, degree: int) -> np.ndarray:
    r'''Finds the array and returns it without h_0, which is always 1.

    Parameters
    ----------
    alpha : float
        see `generate_ar_coefficients`.

    degree : int
        The length that the input needs to be.

    Returns
    -------
    np.ndarray : the coefficents of a power-law autoregressive model

    The coefficients are in reverse order. The array's format is
        [h[degree], h[degree-1], ... h[2], h[1]].

    '''

    H = np.zeros(degree)
    for i, h in zip(range(degree-1, -1, -1), generate_ar_coefficients(alpha)):
        H[i] = h

    return H


def compute(white_noise: np.ndarray,
            coefficients: np.ndarray,
            y: np.ndarray = None) -> np.ndarray:
    r'''Fill `y' with power law noise using `coefficients` & `white_noise`.

    Parameters
    ----------
    white_noise : np.ndarray
        The history of Gaussian noise that to convolve with `coefficients`.
    coefficients : np.ndarray
        This should be in the format computed by `coefficient_array`.
    y : np.ndarray, optional
        Should have a length equal to `len(white_noise) + len(coefficients)`.

    Returns
    -------
    y : np.ndarray
        the last `len(white_noise)` elements of y contain the power-law noise.

    '''

    N = len(white_noise)
    degree = len(coefficients)
    out_length = N + degree

    if y is not None:
        if len(y) != out_length:
            raise ValueError('The output array `y` is the wrong size.')
    else:
        y = np.zeros(out_length)

    y[degree:] = white_noise

    for i in range(N):
        j = i + degree
        y[j] -= np.dot(y[i:j], coefficients)

    return y


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
        self._h = -coefficient_array(self._a, self._d)
        self._buffer = np.zeros(0, dtype=float)

    @property
    def alpha(self) -> float:
        return self._a

    @property
    def degree(self) -> int:
        return self._d

    @property
    def terms(self) -> np.ndarray:
        return np.r_[1.0, -self._h[::-1]]

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


class Noisemaker:
    r'''Holds a buffer for computing AR-based power-law noise.

    Parameters
    ----------
    buffer_size : int
        The amount of memory to reserve for computation.

    Examples
    --------
    >>> nm = Noisemaker(0)
    >>> red = PowerLawNoise(-2.0, 4)
    >>> nm(red, np.linspace(1, 0, 5))
    array([1.  , 1.75, 2.25, 2.5 , 2.5 ])
    >>> nm(red, np.linspace(1, 0, 5), degree=1)
    array([1.  , 1.75, 2.25, 2.5 , 2.5 ])

    '''

    def __init__(self, buffer_size: int):
        self._buffer = np.zeros(buffer_size)

    @property
    def _pad_value(self) -> float:
        r'''The value that means no computations yet.

        Returns
        -------
        float : defaults to zero but could be overridden in subclasses.

        '''

        return 0.0

    def _prepad(self, amount: int) -> np.ndarray:
        r'''Create the prehistory values that go before a noise input.

        Parameters
        ----------
        amount : int
            However many pad values that are needed.

        Returns
        -------
        np.ndarray : `amount` values, defaults to repeating `_pad_value`.

        '''

        return np.repeat(self._pad_value, amount)

    def __call__(self,
                 law: PowerLawNoise,
                 white_noise: np.ndarray,
                 degree: int = None) -> np.ndarray:
        r'''Create power law noise for each term in `white_noise`.

        Parameters
        ----------
        law : PowerLawNoise
            Holds the model's degree and exponent
        white_noise : np.ndarray
            Should be a sample of values from the Standard Normal Distribution
        degree : int, optional
            The caller could use fewer degrees than the law contains.

        Returns
        -------
        power_law_noise : np.ndarray
            The shape of the output's power spectrum will be frequency^alpha.

        '''

        if degree is None:
            degree = law.degree

        noise_length = len(white_noise)
        required_length = degree + noise_length

        if len(self._buffer) < required_length:
            self._buffer = self._prepad(required_length)
        else:
            self._buffer[:degree] = self._prepad(degree)

        self._buffer[-noise_length:] = white_noise
        buffer_length = len(self._buffer)
        for i in range(buffer_length - required_length,
                       buffer_length - degree):
            j = i + degree
            self._buffer[j] += law(self._buffer[i:j])

        return self._buffer[-noise_length:]


class ExamineDegrees:
    def __init__(self, alpha: float, max_degrees: int):
        self._alpha = alpha
        self._make_models(alpha, max_degrees)
        self._y = np.zeros(None)

    def _make_models(self, alpha, max_degrees):
        self._h = coefficient_array(alpha, max_degrees)

    def _y_shape_should_be(self, noise_length: int) -> Tuple[int, int]:
        return (self.N, noise_length + self.N)

    def _set_up_y(self, white_noise):
        shape_should_be = self._y_shape_should_be(len(white_noise))
        if self._y.shape != shape_should_be:
            self._y = np.zeros(shape_should_be)

        self._y[:, self.N:] = white_noise

    @property
    def N(self) -> int:
        return len(self._h)

    @property
    def alpha(self) -> float:
        return self._alpha

    def __call__(self, white_noise_input: np.ndarray) -> np.ndarray:

        self._set_up_y(white_noise_input)

        for d in range(self.N):
            degree = d + 1
            h = self._h[-degree:]
            compute(white_noise_input, h, self._y[d])

        return self._y[:, self.N:]


pln_log_2 = np.log10(2.0)


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
        self._frequencies = np.log10(np.arange(1, fft_size//2))

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
        y[0] -= pln_log_2
        y[-1] -= pln_log_2
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

        spectrum = self.spectrum(noise)[1:-1]
        if len(spectrum) != len(self._frequencies):
            self._make_frequencies(len(noise))
        covariance = np.cov(self._frequencies, spectrum)
        return covariance[0, 1] / covariance[0, 0]


if __name__ == '__main__':

    alphas = 0.01*np.arange(-201, 201)
    degrees = np.arange(1, 65)
    fft_sizes = 2**np.arange(6, 13)

    noisy = Noisemaker(fft_sizes[-1])
    slopy = SpectralSlopeFinder()

    rg = np.random.default_rng(42)

    with open('power_law_output.csv', 'w') as fh:
        fh.write('Power,Degree,Size,Slope\n')

        for N in fft_sizes:
            for alpha in alphas:
                law = PowerLawNoise(alpha, degrees[-1])
                for degree in degrees:
                    for repeat in range(8):
                        m = slopy(noisy(law,
                                        rg.standard_normal(N),
                                        degree))
                        fh.write(f'{alpha},{degree},{N},{m}\n')

    betas = dict([(N, dict([(k,
                             np.array([slope(demo(example,
                                                  np.r_[1, np.zeros(N)],
                                                  example.degree))
                                       for example in [pln.PowerLawNoise(alpha,
                                                                         k)
                                                       for alpha in alphas]]))
                            for k in np.arange(1,26)]))
                  for N in fft_sizes])

    with open('impulse_responses.csv', 'w') as fh:
        fh.write('Size,Degree,Power,Slope\n')
        for N, v in betas.items():
            for a, (d, b) in zip(alphas, v.items()):
                fh.write(f'{N},{d},{a},{b}\n')
