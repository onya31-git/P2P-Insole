# トランスフォーマーの学習用コード
#
# 各処理にsaccessfullyを追加
# コマンドラインから使用するモデルを変更できるようにする
#
import pandas as pd
import numpy as np
import argparse
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from processor.loader import get_datapath_pairs, load_and_combine_data, restructure_insole_data, load_config,  PressureSkeletonDataset
from processor.model import Transformer_Encoder, Skeleton_Loss, train_Transformer_Encoder

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

    # 微分処理を行うならば
    # calculate_grad()

    # データの分割
    train_pressure, val_pressure, train_IMU, val_IMU, train_skeleton, val_skeleton = train_test_split(
        pressure_lr_df, 
        IMU_lr_df,
        skeleton_df,
        test_size=0.2, 
        random_state=42
    )

    # config内容の表示を行う(もっと見やすくする)
    print("---"*20)
    print("<Model Infomation>")
    print(f"model:{args.model}")
    print(f"num_attention_head: {args.model}")
    print(f"num_encoder_layer: {args.model}")
    print(f"dropout:{config['train']['epochs']}")
    print(f"batch_size:{config['train']['batch_size']}")
    # 'input_dim': input_dim,
    # 'd_model': d_model,
    # 'nhead': nhead,
    # 'num_encoder_layers': num_encoder_layers,
    # 'num_joints': num_joints

    # 各種最終セッティング(モデル、ハイパーパラメータ、損失関数、オプティマイザ、スケジューラ)
    input_dim = pressure_lr_df.shape[1] + IMU_lr_df.shape[1] # 圧力+回転+加速度の合計次元数
    d_model = 512
    nhead = 8
    num_encoder_layers = 8
    num_joints = 21 # skeleton_data.shape[1] // 3  # 3D座標なので3で割る
    dropout = 0.2
    batch_size = 128

    # デバイスの設定
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # データローダーの設定
    train_dataset = PressureSkeletonDataset(train_pressure, train_IMU, train_skeleton)
    val_dataset = PressureSkeletonDataset(val_pressure, val_IMU, val_skeleton)
    
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
        nhead= nhead,
        num_encoder_layers= num_encoder_layers,
        num_joints=num_joints,
        num_dims=3,
        dropout=dropout
    ).to(device)

    # 損失関数、オプティマイザ、スケジューラの設定
    criterion = Skeleton_Loss(alpha=1.0, beta=0.1)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=0.0005,
        weight_decay=0.001,
        betas=(0.9, 0.999)
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=5,
        verbose=True
    )

    # トレーニング実行
    train_Transformer_Encoder(
        model,
        train_loader,
        val_loader,
        criterion,
        optimizer, 
        scheduler,
        num_epochs=200,
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
            'nhead': nhead,
            'num_encoder_layers': num_encoder_layers,
            'num_joints': num_joints
        }
    }
    torch.save(final_checkpoint, './weight/final_skeleton_model.pth')   # ファイル名に日付を含んで毎回記録を残せるようにする
    return


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
