# トランスフォーマーの予測用コード
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
    parser.add_argument('--batch-size', type=int, default=32, help='バッチサイズ')
    parser.add_argument('--model', type=str, default='transformer', required=True, help='モデル名')

    return parser