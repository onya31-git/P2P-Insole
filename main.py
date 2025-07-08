#モデルのコントローラ
#
# # modeごとに動的にコンフィグファイルを指定したい。
# # パラメータをコマンドライン引数から設定できるように変更する
#
import argparse
import importlib as iml
    
def main():
    # processor辞書の作成
    processors = {
        'train': iml.import_module('processor.train'),
        'predict': iml.import_module('processor.predict'),
        'visual': iml.import_module('processor.visualization'),
        # 'evaluation': iml.import_module('processor.evaluation'),
        # 'module': iml.import_module('processor.module'),
    }

    # read main-parser
    parser = argparse.ArgumentParser(description='メイン実行スクリプト')
    subparsers = parser.add_subparsers(dest='mode', required=True)

    # read sub-parser
    for name, module in processors.items():
        subparsers.add_parser(name, parents=[module.get_parser()], add_help=False)

    # read arguments
    arg = parser.parse_args()           # コマンドライン引数を解析
    
    # start
    exep = processors[arg.mode]  # 選択されたキーからクラスを取得
    exep.start(arg)                         # 実際の処理を開始

if __name__ == '__main__':
    main()