# Copyright (C) 2020 by Landmark Acoustics LLC
r'''Carry out the simulations and print the results.'''

import numpy as np

from . import \
    NoiseMaker, \
    SpectralSlopeFinder, \
    PowerLawNoise


def _print_heading(handle, headings, sep=',', le='\n'):
    handle.write(sep.join(headings)+le)


def degree_laws(alphas: np.ndarray, degree: int):
    for alpha in alphas:
        yield PowerLawNoise(alpha, degree)


def slope_examples(noisemaker, slopefinder, powers, degrees, sizes, repeats):
    for N in sizes:
        for alpha in powers:
            law = PowerLawNoise(alpha, degrees[-1])
            for degree in degrees:
                for repeat in repeats:
                    yield slopefinder(noisemaker(law,
                                                 rg.standard_normal(N),
                                                 degree))

class SlopeExample:
    r'''Build a bunch of power law noise and yield their slopes.

    Parameters
    ----------
    powers: np.ndarray
        The 'power's in 'power law noise'
    degrees: np.ndarray
        The different possible degrees of the AR models
    sizes: np.ndarray
        The different possible lengths of the time series
    repeats: int
        The number of times to repeat each combination of parameters.
    '''

    def __init__(self,
                 powers: np.ndarray,
                 degrees: np.ndarray,
                 sizes: np.ndarray,
                 repeats: int):
        super().__init__()
        self._alphas = powers
        self._Ks = degrees
        self._Ns = sizes
        self._nm = NoiseMaker(self._Ns[-1])
        self._sf = SpectralSlopeFinder()
        self._reps=repeats

    def make_some_noise(self,
                        size: int) -> np.ndarray:
        return np.r_[1, np.zeros(size)]

    def make_power_law_noise(self,
                             size,
                             degree) -> np.ndarray:
        return self._nm(NoiseProducerMixin.make_some_noise(size),
                        degree)

    def __iter__(self):
        big_K = max(self._Ks)
        reps=range(self._reps)
        original_law = self._nm.law

        for N in self._Ns:
            for alpha in self._alphas:
                self._nm.law = PowerLawNoise(alpha, big_K)
                for K in self._Ks:
                    for it in reps:
                        yield (alpha,
                               K,
                               N,
                               self.make_power_law_noise(N, K))

        self._nm.law = original_law


if __name__ == '__main__':

    alphas = np.linspace(-2, 2, 401)
    degrees = np.arange(1, 65)
    fft_sizes = 2**np.arange(6, 13)

    rg = np.random.default_rng(42)

    exemplar = SlopeExample(alphas,
                            degrees,
                            fft_sizes,
                            8)

    with open('power_law_output.csv', 'w') as fh:
        _print_heading(fh, ['Power', 'Degree', 'Size', 'Slope'])
        for alpha, K, N, noise in exemplar:
            m = slopy(noise)
            fh.write(f'{alpha},{K},{N},{m}\n')

    betas = {N: {degree: np.array([slopy(noisy(law,
                                               np.r_[1, np.zeros(N)],
                                               law.degree))
                           for law in degree_laws(alphas, degree)])
                 for degree in np.arange(1,26)}
             for N in fft_sizes}

    with open('impulse_responses.csv', 'w') as fh:
        _print_heading(fh, ['Size', 'Degree', 'Power', 'Slope'])
        for N, v in betas.items():
            for a, (d, b) in zip(alphas, v.items()):
                fh.write(f'{N},{d},{a},{b}\n')
