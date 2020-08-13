# Copyright (C) 2020 by Landmark Acoustics LLC

from argparse import \
    ArgumentParser, \
    ArgumentError, \
    Namespace

import pytest

from pypowerlawnoise.args import Args


@pytest.fixture(scope='function')
def parser():
    return Args()


def test_default_values(parser, monkeypatch):
    monkeypatch.setattr('sys.argv', [])
    parser()
    expected_sizes = [2**i for i in range(6, 13)]
    expected_degrees = None
    expected_alpha_count = 101
    assert parser.args == Namespace(sizes=expected_sizes,
                                    degrees=expected_degrees,
                                    n_alphas=expected_alpha_count)
                                    

def test_short_args(parser):
    parser('-n', 19,
           '-K', 1, 3, 5, 8,
           '-N', 5, 10, 15)
    assert parser.args == Namespace(sizes=[5, 10, 15],
                                    degrees=[1, 3, 5, 8],
                                    n_alphas=19)

def test_long_arguments(parser):
    parser('--sizes', 3, 6, 9, 12,
           '--degrees', 1, 1, 2, 3,
           '--n_alphas', 42)
    assert parser.args.sizes == [3, 6, 9, 12]
    assert parser.args.degrees == [1, 1, 2, 3]
    assert parser.args.n_alphas == 42


def test_usage(parser):
    assert parser.usage == \
        'usage: __main__.py [-h]' + \
        ' [-N SIZES [SIZES ...]]' + \
        ' [-K DEGREES [DEGREES ...]]' + \
        ' [-n N_ALPHAS]' + \
        '\n'


@pytest.fixture
def mock_argparse_exit(monkeypatch):
    def mock_exit(*args, **kwargs):
        pass

    monkeypatch.setattr(ArgumentParser, 'exit', mock_exit)


@pytest.mark.parametrize('bad_arg', [3.14, 'foo'])
@pytest.mark.parametrize('parameter', ['-N', '-K', '-n'])
def test_type_checking(parser, mock_argparse_exit, parameter, bad_arg):
    with pytest.raises(ArgumentError):
        parser(parameter, bad_arg)


def test_only_one_n_alphas(parser, mock_argparse_exit):
    with pytest.raises(ArgumentError):
        parser('-n', 3, 1, 4)
