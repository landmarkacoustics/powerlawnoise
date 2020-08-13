# Copyright (C) 2020 by Landmark Acoustics LLC
r'''Command-line arguments for when the module is run as a program.'''

import argparse

import numpy as np


class Args:
    def __init__(self):
        parser = argparse.ArgumentParser(description=
                                         'Simulate power law noise')
        parser.add_argument('-N', '--sizes',
                            type=int, nargs='+',
                            default=[2**i for i in range(6, 13)],
                            help='Time series lengths for the simulation.')
        parser.add_argument('-K', '--degrees',
                            type=int, nargs='+',
                            default=None,
                            help='AR model degrees for the simulation.')
        parser.add_argument('-n', '--n_alphas',
                            type=int,
                            default=401,
                            help='How many values of alpha, in [-2, 2].')
        self._parser = parser
        self.args = None

    def __call__(self):
        self.args = self._parser.parse_args()


