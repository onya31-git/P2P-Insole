# 学習処理用プロセッサー
#
# 各処理にsaccessfullyを追加
# コマンドラインから使用するモデルを変更できるようにする
# クオータニオンを使用する
#
import pandas as pd
import numpy as np
import argparse
import joblib
import torch
from torch.utils.data import DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from processor.loader import get_datapath_pairs, load_and_combine_data, restructure_insole_data, calculate_grad, load_config,  PressureSkeletonDataset
from processor.util import print_config
from processor.model import Transformer_Encoder, Skeleton_Loss, train_Transformer_Encoder

def start(args):
    # Load YAML file
    config = load_config(args, args.config, args.model)

    # Set data path
    skeleton_dir = config["location"]["data_path"] + "/skeleton/"
    insole_dir   = config["location"]["data_path"] + "/Insole/"
    
    # Preprocess skeleton data and insole data
    # Load the data and combine the left and right insole data, then separate it into pressure data and IMU data.
    skeleton_insole_datapath_pairs = get_datapath_pairs(skeleton_dir, insole_dir)                           # 骨格データ、insole dataのデータペアを所得
    skeleton_df, insole_left_df, insole_right_df  = load_and_combine_data(skeleton_insole_datapath_pairs)   # 骨格データ、insole data右、insole data左をそれぞれまとめる
    pressure_lr_df, IMU_lr_df = restructure_insole_data(insole_left_df, insole_right_df)                    # insole dataを圧力データとIMUデータに分離、insole dataの左右を結合する
    if config["train"]["use_gradient_data"] == True: calculate_grad()                                       # 微分データの追加(実験用)
    # input_feature_np = np.concatenate([pressure_lr_df, IMU_lr_df], axis=1)

    # Sprit data
    # Skeletal data, pressure data, and IMU data are each split 8:2
    train_pressure, val_pressure, train_IMU, val_IMU, train_skeleton, val_skeleton = train_test_split(
        pressure_lr_df, 
        IMU_lr_df, 
        skeleton_df,
        test_size=0.2,
        shuffle=False
    )

    # Initialize scaler
    pressure_normalizer   = MinMaxScaler()
    imu_normalizer        = MinMaxScaler()
    skeleton_scaler       = MinMaxScaler()

    # Fit the scaler on the training data and transform
    train_pressure_scaled = pressure_normalizer.fit_transform(train_pressure)  # 訓練用圧力データ(fit + transform)
    train_IMU_scaled      = imu_normalizer.fit_transform(train_IMU)            # 訓練用IMUデータ(fit + transform)
    train_skeleton_scaled = skeleton_scaler.fit_transform(train_skeleton)      # 訓練用骨格データ(fit + transform)
    val_pressure_scaled   = pressure_normalizer.transform(val_pressure)        # 検証用圧力データ(fit)
    val_IMU_scaled        = imu_normalizer.transform(val_IMU)                  # 検証用IMUデータ(fit)      
    val_skeleton_scaled   = skeleton_scaler.transform(val_skeleton)            # 検証用骨格データ(fit)  

    # save scaler
    # When I predict the model, I need to use same scaler.
    joblib.dump(skeleton_scaler, './scaler/skeleton_scaler.pkl')

    # combine pressure data and IMU data
    train_input_feature = np.concatenate([train_pressure_scaled, train_IMU_scaled], axis=1)
    val_input_feature   = np.concatenate([val_pressure_scaled, val_IMU_scaled], axis=1) 

    # set final train parameters----------------------------------------------------------------------------
    parameters = {                                                      # parameters 長い、parmsで良くね?
        # model
        "d_model"            : config["train"]["d_model"],
        "n_head"             : config["train"]["n_head"],
        "num_encoder_layer"  : config["train"]["num_encoder_layer"],    # numなのかnなのか統一した方がいい、 他の記号も同様に(d = dim, n = num, l = len)
        "dropout"            : config["train"]["dropout"],

        # learning
        "num_epoch"          : config["train"]["epoch"],
        "batch_size"         : config["train"]["batch_size"],

        # optimize
        "learning_rate"      : config["train"]["learning_rate"],
        "weight_decay"       : config["train"]["weight_decay"],
        
        # loss function
        "loss_alpha"         : config["train"]["loss_alpha"],
        "loss_beta"          : config["train"]["loss_beta"],

        # others
        "use_gradient_data"  : config["train"]["use_gradient_data"],    # gradでいい
        "sequence_len"       : config["train"]["sequence_len"],
        "input_dim"          : pressure_lr_df.shape[1] + IMU_lr_df.shape[1], # 圧力+回転+加速度の合計次元数
        "num_joints"         : skeleton_df.shape[1] // 3,                    # 3D座標なので3で割る
        "num_dims"           :  3
    }
    # <debug> print train parameters
    print_config(parameters)
    #-------------------------------------------------------------------------------------------------

    # set device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # make dataset
    train_dataset = PressureSkeletonDataset(train_input_feature, train_skeleton_scaled, sequence_length=parameters["sequence_len"])
    val_dataset = PressureSkeletonDataset(val_input_feature, val_skeleton_scaled, sequence_length=parameters["sequence_len"])
    
    # set dataloader
    train_loader = DataLoader(
        train_dataset,
        batch_size=parameters["batch_size"],
        shuffle=False,               
        num_workers=4,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=parameters["batch_size"],
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    # モードごとにMLモデルを分岐させる
    # if(args.model == "transformer_encoder"):

    # initialize model
    model = Transformer_Encoder(
        input_dim          = parameters["input_dim"],
        d_model            = parameters["d_model"],
        nhead              = parameters["n_head"],
        num_encoder_layers = parameters["num_encoder_layer"],
        num_joints         = parameters["num_joints"],
        num_dims           = parameters["num_dims"],
        dropout            = parameters["dropout"]
    ).to(device)

    # set loss function
    criterion = Skeleton_Loss(alpha=parameters["loss_alpha"], beta=parameters["loss_beta"])

    # set optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr           = parameters["learning_rate"],
        weight_decay = parameters["weight_decay"],
        betas        = (0.9, 0.999)
    )
    # set scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode     = 'min',
        factor   = 0.5,
        patience = 5,
    )

    # start model training
    train_Transformer_Encoder(
        model,
        train_loader,
        val_loader,
        criterion,
        optimizer, 
        scheduler,
        num_epochs  = parameters["num_epoch"],
        save_path   = './weight/best_skeleton_model.pth',          # ファイル名に日付とモデル名を含んで毎回記録を残せるようにする
        device      = device
    )

    # keep checkpoint
    final_checkpoint = {
        'model_state_dict'      : model.state_dict(),
        'optimizer_state_dict'  : optimizer.state_dict(),
        'scheduler_state_dict'  : scheduler.state_dict(),
        'model_config': {
            'input_dim'         : parameters["input_dim"],
            'd_model'           : parameters["d_model"],
            'nhead'             : parameters["n_head"],
            'num_encoder_layers': parameters["num_encoder_layer"],
            'num_joints'        : parameters["num_joints"]
        }
    }
    torch.save(final_checkpoint, './weight/final_skeleton_model.pth')   # ファイル名に日付とモデル名を含んで毎回記録を残せるようにする
    return


@staticmethod
def get_parser(add_help=False):
    parser = argparse.ArgumentParser(add_help=add_help, description='Training Processor')

    # 基本設定
    parser.add_argument('--model', choices=['transformer_encoder','transformer', 'BERT'], default='transformer_encoder', help='モデル選択')
    parser.add_argument('--config', type=str, default=None, help='YAMLファイルのパス')
    parser.add_argument('--data_path', type=str, default=None)
    parser.add_argument('--sequence_len', type=int, default=None)
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
