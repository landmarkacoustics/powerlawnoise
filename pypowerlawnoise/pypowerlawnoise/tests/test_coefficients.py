# Copyright (C) 2020 by Landmark Acoustics LLC

import pytest

import numpy as np
from numpy.testing import assert_array_almost_equal

from pypowerlawnoise.coefficients import \
    generate_ar_coefficients, \
    coefficient_array


@pytest.mark.parametrize('alpha, coefs',
                         zip([-2, 0, 2],
                             [np.r_[1, -1, 0, 0, 0],
                              np.r_[1, 0, 0, 0, 0],
                              np.r_[1, 1, 1, 1, 1]]),
                         ids=['red', 'white', 'violet'])
def test_trivial_cases(alpha, coefs):
    N = len(coefs) - 1
    for i, h in zip(range(N), generate_ar_coefficients(alpha)):
        assert h == coefs[i]

    assert_array_almost_equal(coefficient_array(alpha, N),
                              coefs)
