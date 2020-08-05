# Copyright (C) 2020 by Landmark Acoustics LLC
r'''An iterator factory that reuses a block of memory to pad inputs.

'''

import numpy as np


class NoiseIterator:
    r'''Holds a buffer for computing AR-based power-law noise.

    This is an iterator factory class that holds a buffer for stepping through
    arrays while pre-padding the array with some "no data" value.

    Parameters
    ----------
    buffer_size : int
        The amount of memory to reserve for computation.
    pad_value : float, optional
        The "no data" value for calculating early values. Defaults to zero.

    Examples
    --------
    >>> ni = NoiseIterator(0)
    >>> ni.buffer_length
    0
    >>> for x in ni.iterate(np.linspace(1, 0, 5), 4):
    ...     x
    ...
    array([0., 0., 0., 1.])
    array([0.  , 0.  , 1.  , 0.75])
    array([0.  , 1.  , 0.75, 0.5 ])
    array([1.  , 0.75, 0.5 , 0.25])
    array([0.75, 0.5 , 0.25, 0.  ])
    >>> ni.buffer_length
    9

    See Also
    --------
    numpy.ndarray : the internal buffer's class

    '''

    def __init__(self, buffer_size: int, pad_value: float = 0.0):
        self._buffer = np.zeros(buffer_size)
        self._pad_value = pad_value

    @property
    def buffer_length(self) -> int:
        r'''The total length of the internal buffer.

        Returns
        -------
        int : the current length of internal buffer.

        '''
        return len(self._buffer)

    def enlarge_perhaps(self, new_buffer_size: int):
        r'''Resize the internal buffer if it is too small.

        Parameters
        ----------
        new_buffer_size: int
           The new size.

        Returns
        -------
        bool : True if the buffer was resized to be longer

        '''

        if self.buffer_length < new_buffer_size:
            self._buffer = np.repeat(self._pad_value, new_buffer_size)
            return True

        return False

    def iterate(self, input_series: np.ndarray, degree: int) -> np.ndarray:
        r'''Create power law noise for each term in `input_series`.

        Parameters
        ----------
        input_series : np.ndarray
            The input noise that some AR model will use.
        degree : int
            The degree of the AR model

        Yields
        ------
        np.ndarray : A single chunk of sound for an AR model to use.

        '''

        input_length = len(input_series)
        required_length = degree + input_length

        if not self.enlarge_perhaps(required_length):
            self._buffer[:degree] = self._pad_value

        stop_index = self.buffer_length - input_length

        self._buffer[stop_index:] = input_series

        start_index = stop_index - degree

        while stop_index < self.buffer_length:
            start_index += 1
            stop_index += 1
            yield self._buffer[start_index:stop_index]
