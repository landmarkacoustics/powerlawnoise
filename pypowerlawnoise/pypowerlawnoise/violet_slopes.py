# Copyright (C) 2020 by Landmark Acoustics LLC
r'''Look at the effect of degree and subspectrum on the slope of violet noise.

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
m = np.array([ssf(n) for n in y])
s = np.array([ssf.spectrum(n) for n in y])
