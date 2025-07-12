# トランスフォーマーの学習用コード
#
# 各処理にsaccessfullyを追加
# コマンドラインから使用するモデルを変更できるようにする
#
import pandas as pd
import numpy as np
import argparse
import time
import torch
from torch.utils.data import DataLoader
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from scipy.ndimage import gaussian_filter1d
from sklearn.model_selection import train_test_split
from processor.loader import get_datapath_pairs, load_and_combine_data, restructure_insole_data, calculate_grad, load_config,  PressureSkeletonDataset
from processor.util import print_config
from processor.model import Transformer_Encoder, Skeleton_Loss, train_Transformer_Encoder

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
    if config["train"]["use_gradient_data"] == True: calculate_grad()                                       # 微分データの追加(実験用)

    # # 正規化と標準化のスケーラー初期化 → 前処理用のコード内に移動
    # pressure_normalizer = MinMaxScaler()
    # rotation_normalizer = MinMaxScaler()

    # pressure_standardizer = StandardScaler(with_mean=True, with_std=True)
    # rotation_standardizer = StandardScaler(with_mean=True, with_std=True)
    
    # # NaN値を補正
    # pressure_lr_df = pressure_lr_df.fillna(0.0)
    # IMU_lr_df = IMU_lr_df.fillna(0.0)
    # skeleton_df = skeleton_df.fillna(0.0)

    # # データの正規化と標準化
    # pressure_lr_df = pressure_standardizer.fit_transform(
    #     pressure_normalizer.fit_transform(pressure_lr_df)
    # )
    # IMU_lr_df = rotation_standardizer.fit_transform(
    #     rotation_normalizer.fit_transform(IMU_lr_df)
    # )

    # sigma=2
    # skeleton_df = skeleton_df.apply(lambda x: gaussian_filter1d(x, sigma=sigma))

    input_feature_np = np.concatenate([pressure_lr_df, IMU_lr_df], axis=1)

    # データの分割
    train_input_feature, val_input_feature, train_skeleton, val_skeleton = train_test_split(
        input_feature_np, 
        skeleton_df,
        test_size=0.2, 
        random_state=42
    )

    # 各種最終セッティング----------------------------------------------------------------------------
    # モデルパラメータ
    d_model = config["train"]["d_model"]
    n_head = config["train"]["n_head"]
    num_encoder_layer = config["train"]["num_encoder_layer"]
    dropout = config["train"]["dropout"]
    sequence_len = config["train"]["sequence_size"]

    # 学習パラメータ
    num_epoch = config["train"]["epoch"]
    batch_size = config["train"]["batch_size"]

    # 最適化パラメータ
    learning_rate = config["train"]["learning_rate"]
    weight_decay = config["train"]["weight_decay"]
    
    # 損失関数パラメータ
    loss_alpha = config["train"]["loss_alpha"]
    loss_beta = config["train"]["loss_beta"]

    # その他のセッティング
    input_dim = pressure_lr_df.shape[1] + IMU_lr_df.shape[1] # 圧力+回転+加速度の合計次元数
    num_joints = skeleton_df.shape[1] // 3  # 3D座標なので3で割る
    num_dims = 3
    #-------------------------------------------------------------------------------------------------

    # <デバッグ>設定内容の表示
    print_config(config, input_dim, num_joints, num_dims)

    # デバイスの設定
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # データローダーの設定
    train_dataset = PressureSkeletonDataset(train_input_feature, train_skeleton, sequence_length=sequence_len)
    val_dataset = PressureSkeletonDataset(val_input_feature, val_skeleton, sequence_length=sequence_len)
    
    # データロード
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    # モードごとにMLモデルを分岐させる

    # モデルの初期化
    model = Transformer_Encoder(
        input_dim= input_dim, # input_dim,
        d_model= d_model,
        nhead= n_head,
        num_encoder_layers= num_encoder_layer,
        num_joints=num_joints,
        num_dims=num_dims,
        dropout=dropout
    ).to(device)

    # 損失関数、オプティマイザ、スケジューラの設定
    criterion = Skeleton_Loss(alpha=1.0, beta=0.1)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
        betas=(0.9, 0.999)
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=5,
    )

    # トレーニング実行
    train_Transformer_Encoder(
        model,
        train_loader,
        val_loader,
        criterion,
        optimizer, 
        scheduler,
        num_epochs=num_epoch,
        save_path='./weight/best_skeleton_model.pth',
        device=device
    )

    # モデルの保存(ファイル名に日付を含めるようにする)
    final_checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'model_config': {
            'input_dim': input_dim,
            'd_model': d_model,
            'nhead': n_head,
            'num_encoder_layers': num_encoder_layer,
            'num_joints': num_joints
        }
    }
    torch.save(final_checkpoint, './weight/final_skeleton_model.pth')   # ファイル名に日付を含んで毎回記録を残せるようにする
    return


@staticmethod
def get_parser(add_help=False):
    parser = argparse.ArgumentParser(add_help=add_help, description='Training Processor')

    # 基本設定
    parser.add_argument('--model', choices=['transformer_encoder','transformer', 'BERT'], default='transformer_encoder', help='モデル選択')
    parser.add_argument('--config', type=str, default=None, help='YAMLファイルのパス')
    parser.add_argument('--data_path', type=str, default=None)
    parser.add_argument('--sequence_size', type=str, default=None)
    parser.add_argument('--use_gradient_data', type=str, default=None)

    # モデルパラメータ
    parser.add_argument('--d_model', type=int, default=None)
    parser.add_argument('--n_head', type=int, default=None)
    parser.add_argument('--num_encoder_layer', type=int, default=None)
    parser.add_argument('--dropout', type=float, default=None)

    # 学習パラメータ
    parser.add_argument('--epoch', type=int, default=None)
    parser.add_argument('--batch_size', type=int, default=None)

    # 最適化パラメータ
    parser.add_argument('--learning_rate', type=float, default=None)
    parser.add_argument('--weight_decay', type=float, default=0.001)

    # 損失関数パラメータ
    parser.add_argument('--loss_alpha', type=float, default=None)
    parser.add_argument('--loss_beta', type=float, default=0.1)

    # Processor
    # feeder
    # model

    return parser
