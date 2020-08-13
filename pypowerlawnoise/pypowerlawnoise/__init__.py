# Copyright (C) 2020 by Landmark Acoustics LLC
r'''pypowerlawnoise

Use python to make power law noise.

'''

from .degree_sequence import degree_sequence
from .power_law_noise import PowerLawNoise
from .spectral_slope_finder import SpectralSlopeFinder

__all__ = [
    'degree_sequence',
    'PowerLawNoise',
    'SpectralSlopeFinder',
]
