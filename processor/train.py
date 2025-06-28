# トランスフォーマーの学習用コード
#
#
#
#
import argparse

from utils.config_loader import load_config

def start(args):

    # YAML設定読み込み
    # config = load_config(args.config)

    # print(config)






@staticmethod
def get_parser(add_help=False):
    parser = argparse.ArgumentParser(add_help=add_help, description='TBase Processor')
    parser.add_argument('--model', choices=['transformer', 'BERT'], default='transformer', help='モデル選択')
    parser.add_argument('--config', type=str, default='config/transformer/train.yaml', help='YAMLファイルのパス')
    parser.add_argument('--data_path', type=str, default='data/training_data')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--save_model', action='store_true')

    # Processor
    # feeder
    # model

    # parameter

    return parser
