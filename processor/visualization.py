# 可視化用コード
#
#
#
#
import argparse

@staticmethod
def get_parser(add_help=False):
    parser = argparse.ArgumentParser(add_help=add_help, description='Transformer Base Processor')

    parser.add_argument('-w', '--work_dir', default='', help='the work folder for storing results')
    parser.add_argument('-c', '--config', default=None, help='path to the configuration file')

    # Processor
    # feeder
    # model

    return parser