# Copyright (C) 2020 by Landmark Acoustics LLC

import pytest

import numpy as np

from pypowerlawnoise import PowerLawNoise

ALPHAS = [-2, -1, 0, 1, 2]
COLOR_NAMES = ['red', 'pink', 'white', 'blue', 'violet']
N = 5
EQUIS = np.linspace(0, 1, N, dtype=float)


def pytest_generate_tests(metafunc):
    if metafunc.cls is not None and 'Trivial' in metafunc.cls.__name__:
        metafunc.parametrize('alpha', ALPHAS, ids=COLOR_NAMES)


@pytest.mark.parametrize('alpha, answer',
                         zip(ALPHAS,
                             [1 + 0.75,
                              1.453125,
                              1.0,
                              0.359375,
                              1.0 - 0.75 - 0.5 - 0.25]),
                         ids=COLOR_NAMES)
def test_integer_powers(alpha, answer):
    law = PowerLawNoise(alpha, N)
    assert law(EQUIS) == answer


class TestTrivialExamples:
    @pytest.mark.parametrize('degree', range(N))
    def test_different_degrees(self, alpha, degree):
        law = PowerLawNoise(alpha, degree)
        assert law.degree == degree
        assert len(law.terms) == degree + 1

    @pytest.mark.parametrize('x', EQUIS)
    def test_zero_degree_is_white_noise(self, alpha, x):
        law = PowerLawNoise(alpha, 0)
        assert law(np.r_[x]) == x
