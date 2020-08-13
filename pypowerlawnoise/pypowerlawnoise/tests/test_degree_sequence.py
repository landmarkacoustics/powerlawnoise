# Copyright (C) 2020 by Landmark Acoustics LLC

import pytest

import numpy as np
from numpy.testing import assert_allclose

from pypowerlawnoise import degree_sequence

@pytest.mark.parametrize('degree, should_be',
                         zip(2**np.arange(6, 13),
                             np.r_[np.c_[1, 2, 3, 6, 11, 20, 35, 64],
                                   np.c_[1, 3, 5, 8, 15, 28, 50, 91],
                                   np.c_[2, 4, 7, 12, 22, 39, 71, 128],
                                   np.c_[3, 5, 9, 17, 30, 55, 100, 181],
                                   np.c_[4, 7, 13, 24, 43, 78, 141, 256],
                                   np.c_[6, 10, 19, 34, 61, 110, 200, 362],
                                   np.c_[8, 14, 26, 48, 86, 156, 283, 512]]))
def test_degree_sequence(degree, should_be):
    assert_allclose(degree_sequence(degree, 8), should_be)
