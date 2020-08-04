# Copyright (C) 2020 by Landmark Acoustics LLC
r'''Compare similar PLN models with different degrees.'''

from typing import Tuple

import numpy as np

from .coefficients import coefficient_array


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
