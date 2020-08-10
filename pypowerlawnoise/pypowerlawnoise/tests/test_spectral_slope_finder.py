# Copyright (C) 2020 by Landmark Acoustics LLC

import pytest
import numpy as np

from numpy.testing import assert_allclose

from pypowerlawnoise import SpectralSlopeFinder


@pytest.mark.parametrize('fft_size',
                         2**np.arange(1, 11))
def test_fft_size_is_correct(fft_size):
    ssf = SpectralSlopeFinder(fft_size)
    freqs = np.arange(1, fft_size//2)
    assert_allclose(ssf.frequencies, np.log10(freqs))


XTRA = np.linspace(-1e-9, 1e-9, 6)

INPUT_SERIES = np.array([np.r_[1, 1, 1, 1, 1, 1] + XTRA,
                         np.r_[1, 0, 1, 0, 1, 0] + XTRA,
                         np.r_[1, 0, 0, 0, 0, 0] + XTRA,
                         np.r_[1, 1, 1, 0, 0, 0] + XTRA,
                         np.r_[1, 0, 0, 1, 0, 0] + XTRA])

SPECTRA = np.array([np.r_[4, 0, 0, 0],
                    np.r_[1, 0, 0, 1],
                    np.r_[1, 2, 2, 1]/9,
                    np.r_[9, 8, 0, 1]/9,
                    np.r_[4, 0, 8, 0]/9,])

SLOPES = np.array([np.nan, np.nan, 0, -60.85359957994146, 59.26863677383678])


@pytest.mark.parametrize('X, should_be',
                         zip(INPUT_SERIES, SPECTRA))
def test_spectrum(X, should_be):
    S = SpectralSlopeFinder.spectrum(X)
    assert_allclose(10**S, should_be, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize('X, should_be',
                         zip(INPUT_SERIES[2:], SLOPES[2:]))
def test_call(X, should_be):
    ssf = SpectralSlopeFinder(len(X))
    spectral_slope = ssf(X)
    assert spectral_slope == should_be
