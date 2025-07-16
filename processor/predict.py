# トランスフォーマーの予測用コード
#
# 各処理にsaccessfullyを追加
# コマンドラインから使用するモデルを変更できるようにする
#
import numpy as np
import pandas as pd
import argparse
import torch
from torch.utils.data import DataLoader
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from scipy.ndimage import gaussian_filter1d
from processor.loader import get_datapath_pairs, load_and_combine_data, restructure_insole_data, load_config, PressureDataset
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

    # スケーラーの初期化
    pressure_normalizer = MinMaxScaler()
    imu_normalizer = MinMaxScaler()
    pressure_standardizer = StandardScaler()
    imu_standardizer = StandardScaler()

    # 訓練データでスケーラーを学習(fit)させ、適用(transform)する
    pressure_scaled = pressure_standardizer.fit_transform(pressure_normalizer.fit_transform(pressure_lr_df))  # 訓練用圧力データ(fit + transform)
    IMU_scaled = imu_standardizer.fit_transform(imu_normalizer.fit_transform(IMU_lr_df))                      # 訓練用IMUデータ(fit + transform)
    input_feature_np = np.concatenate([pressure_scaled, IMU_scaled], axis=1)

    # 最終パラメータ設定
    parameters = {
        # モデルパラメータ
        "d_model": config["predict"]["d_model"],
        "n_head": config["predict"]["n_head"],
        "num_encoder_layer": config["predict"]["num_encoder_layer"],
        "dropout": config["predict"]["dropout"],
        "batch_size": config["predict"]["batch_size"],
        "sequence_len": config["predict"]["sequence_len"],

        # その他のセッティング
        "input_dim": pressure_lr_df.shape[1] + IMU_lr_df.shape[1], # 圧力+回転+加速度の合計次元数    # TransformerEncoderを使用しない場合は圧力とIMUデータの入力場所を別にするため変更する
        "num_joints": skeleton_df.shape[1] // 3,  # 3D座標なので3で割る
        "num_dims":  3,
        "checkpoint_file": config["predict"]["checkpoint_file"] # 動的に指定できるようにする
    }

    # # 微分処理を行うならば
    # calculate_grad()

    # # テストデータセットの作成
    # test_dataset = PressureDataset(input_feature_df)

    # test_loader = DataLoader(
    #     dataset=test_dataset,
    #     batch_size=parameters["batch_size"],
    #     shuffle=False
    # )

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

    # チェックポイントの読み込み
    state_dict = torch.load(parameters["checkpoint_file"], map_location=device, weights_only=True)

    # 問題となっているキーを state_dict から削除(後に必要なくなる)
    if 'positional_encoder.pe' in state_dict:
        del state_dict['positional_encoder.pe']

    # 3. 予測結果を格納するためのリスト
    all_predictions = []

    # 入力データをTensorに変換
    input_tensor = torch.tensor(input_feature_np, dtype=torch.float32).to(device)

    # 修正した state_dict をモデルにロード
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    # # 予測の実行
    # print("Making predictions...")
    # with torch.no_grad():
    #     input_tensor = torch.FloatTensor(input_feature_df).to(device)
    #     input_tensor = input_tensor.unsqueeze(1)  # 形状が [93630, 1, 82]
    #     predictions = model(input_tensor)
    #     predictions = predictions.cpu().numpy()
    # save_predictions(predictions, args.model)

    with torch.no_grad():
        # 4. データローダーからバッチ単位でデータを取り出してループ処理
        for i in range(len(input_tensor) - parameters["sequence_len"]):
            # sequence_lenの長さでシーケンスを切り出す
            sequence = input_tensor[i : i + parameters["sequence_len"]]
            
            # モデルが要求する3D形状 [1, sequence_len, features] に変換
            sequence = sequence.unsqueeze(0)
            
            # モデルで予測を実行
            prediction = model(sequence)
            
            # 結果をリストに保存
            all_predictions.append(prediction.cpu())

    # 5. 全てのバッチの予測結果を一つのテンソルに結合
    final_predictions = torch.cat(all_predictions, dim=0)
    final_predictions_np = final_predictions.numpy()

    print(f"Prediction finished. Output shape: {final_predictions_np.shape}")
    # → 期待される形状: (93630, 出力次元数)

    save_predictions(final_predictions, args.model)

    return 


@staticmethod
def get_parser(add_help=False):
    parser = argparse.ArgumentParser(add_help=add_help, description='Training Processor')

    # 基本設定
    parser.add_argument('--model', choices=['transformer_encoder','transformer', 'BERT'], default='transformer_encoder', help='モデル選択')
    parser.add_argument('--config', type=str, default=None, help='YAMLファイルのパス')
    parser.add_argument('--data_path', type=str, default=None)
    parser.add_argument('--checkpoint_file', type=str, default=None)
    parser.add_argument('--sequence_len', type=str, default=None)

    # モデルパラメータ
    parser.add_argument('--d_model', type=int, default=None)
    parser.add_argument('--n_head', type=int, default=None)
    parser.add_argument('--num_encoder_layer', type=int, default=None)
    parser.add_argument('--dropout', type=float, default=None)
    parser.add_argument('--batch_size', type=str, default=None)

    return parser