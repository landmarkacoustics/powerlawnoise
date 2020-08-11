# Copyright (C) 2020 by Landmark Acoustics LLC
r'''Carry out the simulations and print the results.'''

import numpy as np

from pypowerlawnoise import \
    SpectralSlopeFinder, \
    PowerLawNoise


def _print_heading(handle, headings, sep=',', le='\n'):
    handle.write(sep.join(headings)+le)


if __name__ == '__main__':

    alphas = np.linspace(-2, 2, 401)
    degrees = np.arange(65)
    fft_sizes = 2**np.arange(6, 13)
    slope_finders = {fft_size: SpectralSlopeFinder(fft_size) for fft_size in fft_sizes}
    repeats = 8
    rg = np.random.default_rng(42)

    with open('power_law_output.csv', 'w') as fh:
        _print_heading(fh, ['Power', 'Size', 'Degree', 'Slope'])
        for alpha in alphas:
            law = PowerLawNoise(alpha, degrees[-1])
            for K in degrees:
                for N in fft_sizes:
                    ssf = slope_finders[N]
                    for it in range(repeats):
                        x = rg.standard_normal(N)
                        n = law(x, K)
                        m = ssf(n)
                        fh.write(f'{alpha},{N},{K},{m}\n')
