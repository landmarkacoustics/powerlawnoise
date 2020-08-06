# Copyright (C) 2020 by Landmark Acoustics LLC

import pytest

import numpy as np

from pypowerlawnoise import PowerLawNoise

ALPHAS = [-2, -1, 0, 1, 2]
COLOR_NAMES = ['red', 'pink', 'white', 'blue', 'violet']


@pytest.fixture
def EQUIS():
    return np.linspace(1, 0, 5, dtype=float)


@pytest.fixture(scope='function')
def law(request, EQUIS):
    return PowerLawNoise(request.param, len(EQUIS))


@pytest.mark.parametrize('law, answer',
                         zip(ALPHAS,
                             [0.25, 0.2734375, 0.0, -0.8203125, -2.5]),
                         indirect=['law'],
                         ids=COLOR_NAMES)
def test_integer_powers(law, answer, EQUIS):
    assert law(EQUIS) == answer
