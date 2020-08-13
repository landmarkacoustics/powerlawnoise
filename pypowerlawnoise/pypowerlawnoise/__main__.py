# Copyright (C) 2020 by Landmark Acoustics LLC
r'''Carry out the simulations and print the results.'''


import argparse

from .degree_sequence import degree_sequence

import numpy as np

from pypowerlawnoise import \
    SpectralSlopeFinder, \
    PowerLawNoise


parser = argparse.ArgumentParser(prog='pypowerlawnoise',
                                 description= 'Simulate power law noise')

parser.add_argument('output_file',
                    type=argparse.FileType('w'),
                    help='The output file, which will contain CSV data.')

group = parser.add_argument_group(title='Power Law Noise Parameters',
                                  description='Arguments that are directly'
                                  ' involved in calculating power law noise.')

group.add_argument('-N', '--sizes',
                   type=int, nargs='+',
                   default=[2**i for i in range(6, 13)],
                   help='Time series lengths for the simulation.')

group.add_argument('-K', '--degrees',
                   type=int, nargs='+',
                   default=None,
                   help='AR model degrees for the simulation.')

group.add_argument('-n', '--n_alphas',
                   type=int,
                   default=101,
                   help='How many values of alpha, in [-2, 2].')

group = parser.add_argument_group(title='Simulation Parameters',
                                  description='Arguments that control how each'
                                  ' simulation is implemented.')

group.add_argument('-r', '--repeats',
                   type=int,
                   default=10,
                   help='How many repetitions to run for each combination'
                   ' of parameters')

group.add_argument('--seed',
                   type=int,
                   default=None,
                   help='A seed for the random number generator')

group = parser.add_argument_group(title='Centered Degrees (optional)',
                                  description='Arguments that control how the'
                                  ' script generates degrees centered on the'
                                  ' square root of each size. These are only'
                                  ' used when `-K` is omitted.')

group.add_argument('-d', '--n_degrees',
                   type=int,
                   default=40,
                   help='How many centered degrees to examine')

group.add_argument('-m', '--multiplier',
                   type=float,
                   default=8,
                   help='The range of the centered degrees')

arguments = parser.parse_args()

alphas = np.linspace(-2, 2, arguments.n_alphas)
fft_sizes = arguments.sizes
slope_finders = {fft_size: SpectralSlopeFinder(fft_size)
                 for fft_size in fft_sizes}
repeats = arguments.repeats
rg = np.random.default_rng(arguments.seed)

with arguments.output_file as fh:

    fh.write(','.join(['Power', 'Size', 'Degree', 'Slope']) + '\n')

    for N in fft_sizes:

        for alpha in alphas:

            if arguments.degrees is None:
                degrees = degree_sequence(N,
                                          arguments.n_degrees,
                                          arguments.multiplier)
            else:
                degrees = arguments.degrees

            law = PowerLawNoise(alpha, degrees[-1])

            for K in degrees:

                ssf = slope_finders[N]

                for it in range(repeats):

                    x = rg.standard_normal(N)
                    n = law(x, K)
                    m = ssf(n)
                    fh.write(f'{alpha},{N},{K},{m}\n')
