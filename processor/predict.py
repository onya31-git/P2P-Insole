# トランスフォーマーの予測用コード
#
# 各処理にsaccessfullyを追加
# コマンドラインから使用するモデルを変更できるようにする
#
import numpy as np
import pandas as pd
import argparse
import torch
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from scipy.ndimage import gaussian_filter1d
from processor.loader import get_datapath_pairs, load_and_combine_data, restructure_insole_data, load_config
from processor.model import Transformer_Encoder, save_predictions

def start(args):

    # YAML設定読み込み
    config = load_config(args, args.config, args.model)

    # データパスの指定
    skeleton_dir = config["location"]["data_path"] + "/skeleton/"
    insole_dir   = config["location"]["data_path"] + "/Insole/"
    
    # skeleton dataとinsole data の前処理
    skeleton_insole_datapath_pairs = get_datapath_pairs(skeleton_dir, insole_dir)                           # 骨格データ、insole dataのデータペアを所得
    skeleton_df, insole_left_df, insole_right_df  = load_and_combine_data(skeleton_insole_datapath_pairs)   # 骨格データ、insole data右、insole data左をそれぞれまとめる
    pressure_lr_df, IMU_lr_df = restructure_insole_data(insole_left_df, insole_right_df)                    # insole dataを圧力データとIMUデータに分離、insole dataの左右を結合する

    # 最終パラメータ設定
    parameters = {
        # モデルパラメータ
        "d_model": config["train"]["d_model"],
        "n_head": config["train"]["n_head"],
        "num_encoder_layer": config["train"]["num_encoder_layer"],
        "dropout": config["train"]["dropout"],
        
        # 損失関数パラメータ
        "loss_alpha": config["train"]["loss_alpha"],
        "loss_beta": config["train"]["loss_beta"],

        # その他のセッティング
        "input_dim": pressure_lr_df.shape[1] + IMU_lr_df.shape[1], # 圧力+回転+加速度の合計次元数    # TransformerEncoderを使用しない場合は圧力とIMUデータの入力場所を別にするため変更する
        "num_joints": skeleton_df.shape[1] // 3,  # 3D座標なので3で割る
        "num_dims":  3,
        "checkpoint_file": config["train"]["checkpoint_file"] # 動的に指定できるようにする
    }

    # # 微分処理を行うならば
    # calculate_grad()

    # 正規化と標準化のスケーラー初期化 → 前処理用のコード内に移動
    pressure_normalizer = MinMaxScaler()
    rotation_normalizer = MinMaxScaler()

    pressure_standardizer = StandardScaler(with_mean=True, with_std=True)
    rotation_standardizer = StandardScaler(with_mean=True, with_std=True)
    
    # NaN値を補正
    pressure_lr_df = pressure_lr_df.fillna(0.0)
    IMU_lr_df = IMU_lr_df.fillna(0.0)
    skeleton_df = skeleton_df.fillna(0.0)

    # データの正規化と標準化
    pressure_lr_df = pressure_standardizer.fit_transform(
        pressure_normalizer.fit_transform(pressure_lr_df)
    )
    IMU_lr_df = rotation_standardizer.fit_transform(
        rotation_normalizer.fit_transform(IMU_lr_df)
    )

    sigma=2
    skeleton_df = skeleton_df.apply(lambda x: gaussian_filter1d(x, sigma=sigma))

    # 圧力、IMUデータを結合したデータも用意しておく
    input_feature_df = np.concatenate([pressure_lr_df, IMU_lr_df], axis=1)

    # デバイスの設定
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")


    # 入力時のモデルの入力から分岐させれるようにする

    # モデルの初期化（固定パラメータを使用）
    model = Transformer_Encoder(
        input_dim=parameters["input_dim"], 
        d_model= parameters["d_model"],
        nhead=parameters["n_head"],
        num_encoder_layers=parameters["num_encoder_layer"],
        num_joints=parameters["num_joints"],
        num_dims=parameters["num_dims"],
        dropout=parameters["dropout"]
    ).to(device)

    # チェックポイントの読み込み（weights_only=Trueを追加）
    checkpoint = torch.load(parameters["checkpoint_file"], map_location=device, weights_only=True)

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
    parser = argparse.ArgumentParser(add_help=add_help, description='Training Processor')

    # 基本設定
    parser.add_argument('--model', choices=['transformer_encoder','transformer', 'BERT'], default='transformer_encoder', help='モデル選択')
    parser.add_argument('--config', type=str, default=None, help='YAMLファイルのパス')
    parser.add_argument('--data_path', type=str, default=None)
    parser.add_argument('--checkpoint_file', type=str, default=None)

    # モデルパラメータ
    parser.add_argument('--d_model', type=int, default=None)
    parser.add_argument('--n_head', type=int, default=None)
    parser.add_argument('--num_encoder_layer', type=int, default=None)
    parser.add_argument('--dropout', type=float, default=None)

    # 損失関数パラメータ
    parser.add_argument('--loss_alpha', type=float, default=None)
    parser.add_argument('--loss_beta', type=float, default=0.1)

    return parser