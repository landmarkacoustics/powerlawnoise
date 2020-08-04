# Copyright (C) 2020 by Landmark Acoustics LLC
r'''Memory management for using a `PowerLawNoise` object
'''

import numpy as np

from . import PowerLawNoise


class NoiseMaker:
    r'''Holds a buffer for computing AR-based power-law noise.

    Parameters
    ----------
    buffer_size : int
        The amount of memory to reserve for computation.

    Examples
    --------
    >>> nm = NoiseMaker(0)
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
