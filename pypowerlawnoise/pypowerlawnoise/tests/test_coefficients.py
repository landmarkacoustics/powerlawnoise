# Copyright (C) 2020 by Landmark Acoustics LLC

import pytest

from numpy.testing import assert_array_almost_equal

from pypowerlawnoise.coefficients import \
    generate_ar_coefficients, \
    coefficient_array


def pytest_generate_tests(metafunc):
    trivial_cases = {
        'red': (-2.0,
                [1, 1, 0.0,]),
        'white': (0.0, [1, 0, 0.0,]),
        'violet': (2.0, [1, -1.0, -1.0,])
    }
    if 'trivial' in metafunc.function.__name__:
        metafunc.parametrize('alpha, coefs',
                             trivial_cases.values(),
                             ids=trivial_cases)


def test_ar_generator_on_trivial_cases(alpha, coefs):
    for i, h in zip(range(len(coefs) - 1), generate_ar_coefficients(alpha)):
        assert h == coefs[i]

def test_coef_array_on_trivial_cases(alpha, coefs):
    assert_array_almost_equal(coefficient_array(alpha, len(coefs) - 1),
                              coefs)
