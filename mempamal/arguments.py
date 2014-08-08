# Author: Benoit Da Mota <damota.benoit@gmail.com>
#
# License: BSD 3 clause
"""
Build arguments parser for the scripts (mapper, reducers and command builder).
"""
import argparse


def get_map_argparser():
    """Build command line arguments parser for a mapper.

    Arguments parser compatible with the commands builder workflows.
    """
    parser = argparse.ArgumentParser()

    parser.add_argument("crossval",
                        help="JSON file to configure cross validation scheme")
    parser.add_argument("method",
                        help="JSON file to configure the method")
    parser.add_argument("dataset",
                        help="Joblib file with data and folds")
    parser.add_argument("out",
                        help="Filename to output the results")
    parser.add_argument("outer", type=int,
                        help="Outer CV Id")

    parser.add_argument("--inner", type=int,
                        help="Inner CV Id")

    # verbose mode
    parser.add_argument("-v", "--verbose", help="verbose mode",
                        action="store_true")
    return parser


def get_ired_argparser():
    """Build command line arguments parser for an inner reducer.

    Arguments parser compatible with the commands builder workflows.
    """
    parser = argparse.ArgumentParser()

    parser.add_argument("crossval",
                        help="JSON file to configure cross validation scheme")
    parser.add_argument("method",
                        help="JSON file to configure the method")
    parser.add_argument("dataset",
                        help="Joblib file with data and folds")
    parser.add_argument("out",
                        help="Filename to output the results")
    parser.add_argument("in",
                        help="Filename template for input files")
    parser.add_argument("outer", type=int,
                        help="Outer CV Id")

    # verbose mode
    parser.add_argument("-v", "--verbose", help="verbose mode",
                        action="store_true")
    return parser


def get_ored_argparser():
    """Build command line arguments parser for an outer reducer.

    Arguments parser compatible with the commands builder workflows.
    """
    parser = argparse.ArgumentParser()

    parser.add_argument("out",
                        help="Filename to output the results")
    parser.add_argument("in",
                        help="Filename template for input files")

    # verbose mode
    parser.add_argument("-v", "--verbose", help="verbose mode",
                        action="store_true")
    return parser
