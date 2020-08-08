# Copyright (C) 2020 by Landmark Acoustics LLC

import pytest

import numpy as np
from numpy.testing import assert_allclose

from pypowerlawnoise import \
    PowerLawNoise, \
    SpectralSlopeFinder


class SlopeCase:
    r'''Test objects that depend on FFT size.'''
    def __init__(self, noise):
        self.fft_size = len(noise)
        self.noise = noise
        self.finder = SpectralSlopeFinder(self.fft_size)

    def __repr__(self):
        return f'<A Slope Case with an FFT size of {self.fft_size}>'


@pytest.fixture(params=2**np.arange(8, 12), scope='module')
def slope_case(request, seed: int = None):
    r'''Initialize components that are expensive and depend on FFT size.'''

    if seed is None:
        seed = 42

    rng = np.random.default_rng(seed)
    fft_size = request.param
    return SlopeCase(rng.standard_normal(fft_size))


@pytest.mark.parametrize('alpha', np.linspace(-2, 2, 3),
                         ids=['red', 'white', 'violet'])
def test_slope_works(slope_case, alpha):
    degree = 2 * int(np.sqrt(slope_case.fft_size))
    law = PowerLawNoise(alpha, degree)
    noise = np.array([x for x in law.generate_noise(slope_case.noise)])
    noise *= np.hamming(len(noise))
    assert_allclose(slope_case.finder(noise), alpha, rtol=1e-1, atol=1/3)
