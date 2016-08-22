#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This is predict script. Running it returns the prediction using the selected model

To run this script uncomment the following line in the
entry_points section in setup.cfg:

    console_scripts =
     predict = infodenguepredict.predict:run


"""

import argparse
import sys
import logging

from infodenguepredict import __version__

__author__ = "Flávio Codeço Coelho"
__copyright__ = "Flávio Codeço Coelho"
__license__ = "gpl3"

_logger = logging.getLogger(__name__)


def predict(model, test_data):
    """
    generate prediction from existing model based on test data
    :param model: Trained model saved previously
    :param test_data: data from which to generate the predictions
    :return:

    """
    pass


def parse_args(args):
    """
    Parse command line parameters

    :param args: command line parameters as list of strings
    :return: command line parameters as :obj:`argparse.Namespace`
    """
    parser = argparse.ArgumentParser(
        description="Incidence prediction tool")
    parser.add_argument(
        '--version',
        action='version',
        version='InfoDenguePredict {ver}'.format(ver=__version__))
    parser.add_argument(
        dest="m",
        help="model file",
        type=str,
        metavar="INT")
    parser.add_argument(
        dest="d",
        help="Test data file (CSV)",
        type=str
    )
    parser.add_argument(
        '-v',
        '--verbose',
        dest="loglevel",
        help="set loglevel to INFO",
        action='store_const',
        const=logging.INFO)
    parser.add_argument(
        '-vv',
        '--very-verbose',
        dest="loglevel",
        help="set loglevel to DEBUG",
        action='store_const',
        const=logging.DEBUG)
    return parser.parse_args(args)


def main(args):
    args = parse_args(args)
    logging.basicConfig(level=args.loglevel, stream=sys.stdout)
    _logger.debug("Generating predition...")
    print("The predicted incidence is...")
    _logger.info("Script ends here")


def run():
    main(sys.argv[1:])


if __name__ == "__main__":
    run()
