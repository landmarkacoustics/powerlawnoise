# Copyright (C) 2020 by Landmark Acoustics LLC
r'''Look at the effect of degree and subspectrum on the slope of violet noise.

My conclusions from inspection is that there is no harm in calculating the
slope using all of the spectrum, except for the constant-frequency term.

'''

import . as pln

import numpy as np
import pickle
from matplotlib import pyplot as plt

fft_size = 2048
degrees = np.arange(0, fft_size+1)

rg = np.random.default_rng(42)
x = rg.standard_normal(fft_size)

law = pln.PowerLawNoise(2, fft_size)

y = np.array([law(x, d) for d in degrees])

ssf = pln.SpectralSlopeFinder(fft_size)
half_fft = len(ssf.frequencies)

m = np.array([ssf(n) for n in y])
s = np.array([ssf.spectrum(n) for n in y])

def regression_slope(x, y):
    covariance = np.cov(x, y)
    return covariance[0, 1] / covariance[0, 0]

slopes = np.array([np.fromiter([regression_slope(ssf.frequencies[1:j],
                                               spk[1:j])
                                for j in range(3, half_fft)],
                               float,
                               half_fft - 3)
                   for spk in s])


