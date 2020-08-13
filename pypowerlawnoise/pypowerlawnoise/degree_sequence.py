# Copyright (C) 2020 by Landmark Acoustics LLC
r'''A sequence of degrees centered on the square root of a series' size.'''

import numpy as np


def degree_sequence(N: int,
                    out_length: int = 4,
                    multiplier: float = 8) -> np.ndarray:
    r'''A sequence of degrees centered on the square root of a series' size.

    The degrees will be integers and they will be centered on a log scale.

    Parameters
    ----------
    N : int
        The size of a time series
    out_length : int, optional
        How many different degrees to calculate, at most `N`, defaults to 40.
    multiplier : float, optional
        How far from the square root of the size the sequence should extend.

    Returns
    -------
    np.ndarray : an array of integers, without duplicates, in [1, N].

    '''

    b = 0.5 * np.log10(N)
    bound = np.log10(multiplier)
    logs = np.linspace(b - bound, b + bound, out_length)
    return np.array(np.unique(np.around(10**logs)), dtype=int)
