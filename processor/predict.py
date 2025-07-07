# トランスフォーマーの予測用コード
#
# 各処理にsaccessfullyを追加
# コマンドラインから使用するモデルを変更できるようにする
#
import numpy as np
import pandas as pd
import argparse
from processor.loader import get_datapath_pairs, load_and_combine_data, restructure_insole_data, load_config
from processor.model import Transformer_Encoder, Skeleton_Loss, train_Transformer_Encoder, save_predictions

def start(args):

    # YAML設定読み込み
    config = load_config(args.config)

    # データパスの指定
    skeleton_dir = config["common"]["data_path"] + "/skeleton/"
    insole_dir   = config["common"]["data_path"] + "/Insole/"
    
    # 骨格データ、インソールデータのデータペアを所得
    skeleton_insole_datapath_pairs = get_datapath_pairs(skeleton_dir, insole_dir)

    # 骨格データ、インソールデータ右、インソールデータ左をそれぞれまとめる
    skeleton_df, insole_left_df, insole_right_df  = load_and_combine_data(skeleton_insole_datapath_pairs)

    # インソールデータを圧力データとIMUデータに分離、インソールデータの左右を結合する
    pressure_lr_df, IMU_lr_df = restructure_insole_data(insole_left_df, insole_right_df)

    # 圧力、IMUデータを結合したデータも用意しておく
    input_feature_df = np.concatnate([pressure_lr_df, IMU_lr_df])

    # 最終セッティング
    input_dim = pressure_lr_df.shape[1] + IMU_lr_df.shape[1]            # TransformerEncoderを使用しない場合は圧力とIMUデータの入力場所を別にするため変更する
    num_joints = skeleton_df.shape[1] // 3
    checkpoint_file = './weight/best_skeleton_model_test4Tasks.pth'     # 動的に指定できるようにする

    # デバイスの設定
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")


    # 入力時のモデルの入力から分岐させれるようにする

    # モデルの初期化（固定パラメータを使用）
    model = Transformer_Encoder(
        input_dim=input_dim,
        d_model=512,
        nhead=8,
        num_encoder_layers=8,
        num_joints=num_joints,
        num_dims=3,
        dropout=0.2
    ).to(device)

    # チェックポイントの読み込み（weights_only=Trueを追加）
    checkpoint = torch.load(checkpoint_file, map_location=device, weights_only=True)

    # モデルの重みを読み込み
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # 予測の実行
    print("Making predictions...")
    with torch.no_grad():
        input_tensor = torch.FloatTensor(input_feature_df).to(device)
        predictions = model(input_tensor)
        predictions = predictions.cpu().numpy()

    save_predictions(predictions, args.model)

    return 


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