# Copyright (C) 2020 by Landmark Acoustics LLC
r'''Memory management for using a `PowerLawNoise` object
'''

import numpy as np

from . import PowerLawNoise


class NoiseMaker:
    r'''Holds a buffer for computing AR-based power-law noise.

    This needs to have a buffer because the noise is stateful. Each new input
    changes the future outputs.

    Parameters
    ----------
    buffer_size : int
        The amount of memory to reserve for computation.
    law : PowerLawNoise, optional
        An autoregressive model for generating power law noise. Defaults to
        white noise.

    Examples
    --------
    >>> nm = NoiseMaker(0, PowerLawNoise(-2.0, 4))
    >>> nm(np.linspace(1, 0, 5))
    array([1.  , 1.75, 2.25, 2.5 , 2.5 ])
    >>> nm(np.linspace(1, 0, 5), degree=1)
    array([1.  , 1.75, 2.25, 2.5 , 2.5 ])

    '''

    _pad_value = 0.0

    def __init__(self, buffer_size: int, law: PowerLawNoise = None):
        self._buffer = np.zeros(buffer_size)
        if law is None:
            law = PowerLawNoise(0.0, 0)
        self._law = law

    @property
    def law(self) -> PowerLawNoise:
        r'''An autoregressive model for generating power law noise.'''
        return self._law

    @law.setter
    def law(self, new_law):
        self._law = new_law

    @property
    def pad_value(self) -> float:
        r'''The value that means no computations yet.

        Returns
        -------
        float : defaults to zero but could be overridden in subclasses.

        '''

        return self._pad_value

    def _prepad(self, amount: int) -> np.ndarray:
        r'''Create the prehistory values that go before a noise input.

        Parameters
        ----------
        amount : int
            However many pad values that are needed.

        Returns
        -------
        np.ndarray : `amount` values, defaults to repeating `pad_value`.

        '''

        return np.repeat(self.pad_value, amount)

    def __call__(self,
                 white_noise: np.ndarray,
                 degree: int = None) -> np.ndarray:
        r'''Create power law noise for each term in `white_noise`.

        Parameters
        ----------
        white_noise : np.ndarray
            Should be a sample of values from the Standard Normal Distribution
        degree : int, optional
            The caller could use fewer degrees than the law contains.

        Returns
        -------
        power_law_noise : np.ndarray
            The slope of the output's power spectrum will be frequency^alpha.

        '''

        if degree is None:
            degree = self.law.degree

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
            self._buffer[j] += self._law(self._buffer[i:j])

        return self._buffer[-noise_length:]


def make_some_noise(law, input_source, degree) -> float:
    buf = np.zeros(degree)
    for x in input_source:
        buf.roll(-1)
        buf[-1] = x
        buf[-1] = law(buf)
        yield buf
