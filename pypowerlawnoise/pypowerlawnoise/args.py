# Copyright (C) 2020 by Landmark Acoustics LLC
r'''Command-line arguments for when the module is run as a program.'''

import argparse

class Args:
    r'''Assemble the argument parser for the command-line version.'''

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
                            default=101,
                            help='How many values of alpha, in [-2, 2].')
        self._parser = parser
        self._args = None

    def __call__(self, *args):
        r'''Parse the arguments.

        Parameters
        ----------
        *args : optional positional arguments that are passed to the parser.

        '''
        if args:
            self._args = self._parser.parse_args([str(x) for x in args])
        else:
            self._args = self._parser.parse_args()

    @property
    def args(self) -> argparse.Namespace:
        r'''The results from parsing command-line arguments.

        Returns
        -------
        list : all the arguments, initially `None`.

        See Also
        --------
        argparse : the module that makes this attribute.

        '''
        return self._args

    @property
    def usage(self) -> str:
        r'''Print the usage string from the parser.

        Returns
        -------
        str : the parser's usage string

        '''

        return self._parser.format_usage()
