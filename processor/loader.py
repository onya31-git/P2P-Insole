# load関係の関数
#
#   関数の移動を検討する
#   YAMLファイルが一つでもいいか検証する
#
import pandas as pd
import numpy as np
import os
import glob
import yaml
import time
import torch
from collections import defaultdict
from torch.utils.data import Dataset

def load_config(args,config_path, model):
    """指定されたパスからYAMLファイルを読み込み、Pythonオブジェクトとして返す
    Args:
        path (str): 読み込む設定ファイル（.yaml）のファイルパス
    Returns:
        dict: YAMLファイルの内容から変換された辞書オブジェクト
    """
    # configファイル名が指定されている場合は指定されたパスを、そうでなければモデル名から推測
    if config_path is None:
        config_path = f'config/{model}/{args.mode}.yaml'
    
    # 使用したconfigファイルをプロンプトに表示 
    print(f"<load config>")
    print(f"Loading configuration from: {config_path}")
    time.sleep(0.5)

    # configファイルをロード
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # コマンドライン引数で設定を上書き
    cli_args = vars(args)   # vars()でargsを辞書に変換
    for key, value in cli_args.items():  # 値がNoneでない引数はconfigで上書きする
        if value is not None:           #
            config[key] = value         #

    return config


def get_datapath_pairs(skeleton_dir, insole_dir):
    """指定されたディレクトリから、共通タグを持つskeletonとinsoleのファイルパスをペアリングする
    Args:
        skeleton_dir (str): スケルトンデータ（*_skeleton.csv）が格納されているディレクトリのパス
        insole_dir (str): インソールデータ（*_Insole_*.csv）が格納されているディレクトリのパス
    Returns:
        defaultdict: タグをキーとした辞書。
            値は `{tag:{'skeleton': str, 'insole': list[str]}}` の形式
    """
    # データパスの表示
    print("---"*20)
    print(f"<Dataset Infomation>")
    print(f"skeleton data path : {skeleton_dir}")
    print(f"Inosole data path : {insole_dir}")
    time.sleep(0.5)

    # フォルダ内のcsvファイルを全て所得する
    skeleton_files = glob.glob(os.path.join(skeleton_dir, "*_skeleton.csv"))
    insole_files = glob.glob(os.path.join(insole_dir, "*_Insole_*.csv"))

    # skeletonとInsoleのデータペアを格納する辞書を作成する
    data_pairs = defaultdict(lambda: {'skeleton': None, 'insole': []})

    # skeletonファイルからタグを抽出して辞書に格納する
    for file_path in skeleton_files:
        filename = os.path.basename(file_path)
        tag = filename.replace('_skeleton.csv', '')
        data_pairs[tag]['skeleton'] = file_path
    
    # insoleファイルからタグを抽出して辞書に格納する
    for file_path in insole_files:
        filename = os.path.basename(file_path)
        tag = filename.split('_Insole_')[0]

        # 辞書を参照して対応するskeletonファイルが存在する場合のみ追加する
        if tag in data_pairs:
            data_pairs[tag]['insole'].append(file_path)

    # 抽出結果を表示
    data_i=0
    for tag, paths in data_pairs.items():
        data_i+=1
        print(f"")
        print(f"Data_{data_i}_{tag}")
        print(f"skeleton: {paths['skeleton']}")
        print(*[f"insole: {f}" for f in sorted(paths['insole'])], sep='\n')
        time.sleep(0.3)
    print("---"*20)

    return data_pairs


def load_and_combine_data(data_pairs):
    """ファイルパスの辞書からデータを読み込み、カテゴリ別に結合したDataFrameを返す
    Args:
        data_pairs (dict): タグをキーとし、ファイルパスの辞書を値に持つオブジェクト
    Returns:
        tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: 
    """
    # 各データを格納する配列を作成
    all_skeleton_df     = []
    all_insole_left_df  = []
    all_insole_right_df = []
    
    # 各データをdfに変換してall_kindに格納する
    for tag, paths in data_pairs.items():
        skeleton_df     = pd.read_csv(paths['skeleton'])
        insole_left_df  = pd.read_csv(paths['insole'][0])
        insole_right_df = pd.read_csv(paths['insole'][1])

        all_skeleton_df.append(skeleton_df)
        all_insole_left_df.append(insole_left_df)
        all_insole_right_df.append(insole_right_df)

    return (pd.concat(all_skeleton_df, ignore_index=True),
            pd.concat(all_insole_left_df, ignore_index=True),
            pd.concat(all_insole_right_df, ignore_index=True))


def restructure_insole_data(insole_left_df, insole_right_df):
    """左右のインソールデータフレームを、圧力データとIMUデータに分割・再結合する
    Args:
        insole_left_df (pd.DataFrame): 左足のインソールデータを含むDataFrame
        insole_right_df (pd.DataFrame): 右足のインソールデータを含むDataFrame
    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: 分割・再結合されたDataFrameのタプル
    """
    # 左足データから各種センサー値を抽出
    pressure_left_df = insole_left_df.drop(["Gyro_x","Gyro_y","Gyro_z","Acc_x","Acc_y","Acc_z"],axis=1)
    IMU_left_df      = insole_left_df[["Gyro_x","Gyro_y","Gyro_z","Acc_x","Acc_y","Acc_z"]]

    # 右足データから各種センサー値を抽出
    pressure_right_df = insole_right_df.drop(["Gyro_x","Gyro_y","Gyro_z","Acc_x","Acc_y","Acc_z"],axis=1)
    IMU_right_df      = insole_right_df[["Gyro_x","Gyro_y","Gyro_z","Acc_x","Acc_y","Acc_z"]]

    # 左右データの結合
    pressure_lr = pd.concat([pressure_left_df, pressure_right_df], axis=1)
    IMU_lr      = pd.concat([IMU_left_df, IMU_right_df], axis=1)

    return pressure_lr, IMU_lr


def calculate_grad(pressure_lr, IMU_lr):
    """圧力データとIMUデータに1次および2次微分を計算し、特徴量として結合する。
    Args:
        pressure_lr (np) : 圧力センサーの時系列データ
        IMU_lr (np.) : IMUの時系列データ
    Returns:
        tuple[np, np]: 拡張された特徴量を持つタプル
    """
    # 1次微分と2次微分の計算(使用する場合)
    pressure_grad1 = np.gradient(pressure_lr, axis=0)
    pressure_grad2 = np.gradient(pressure_grad1, axis=0)
    IMU_grad1 = np.gradient(IMU_lr, axis=0)
    IMU_grad2 = np.gradient(IMU_grad1, axis=0)
    pressure_features = np.concatenate([
        pressure_lr,
        pressure_grad1,
        pressure_grad2,
    ], axis=1)
    IMU_features = np.concatenate([
        IMU_lr,
        IMU_grad1,
        IMU_grad2,
    ], axis=1)

    return pressure_features, IMU_features
    

class PressureSkeletonDataset(Dataset):
    """圧力データと骨格データを扱うためのPyTorchカスタムデータセット。
    Args:
        pressure_data (pd): 圧力データのシーケンス。
        skeleton_data (pd): 骨格データのシーケンス。

    Returns:
        pressure_data (torch.Tensor): Tensorに変換された圧力データ。
        skeleton_data (torch.Tensor): Tensorに変換された骨格データ。
    """
    def __init__(self, input_feature, skeleton_data, sequence_length):
        self.sequence_length = sequence_length
        self.input_data = input_feature
        self.skeleton_data = skeleton_data
        
    def __len__(self):
        return len(self.input_data) - self.sequence_length + 1
    
    def __getitem__(self, index):
        # 入力シーケンスの切り出し
        X = self.input_data[index : index + self.sequence_length]
        y = self.skeleton_data[index + self.sequence_length - 1]

        return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)
    

class PressureDataset(Dataset):
    """
    予測用の入力データを扱うためのPyTorchカスタムデータセット。
    
    Args:
        features (np.ndarray): 予測したい入力データ（Numpy配列）
    """
    def __init__(self, features):
        # データをfloat型のPyTorchテンソルに変換
        self.features = torch.tensor(features, dtype=torch.float32)

    def __len__(self):
        # データの総数がそのままデータセットの長さになります
        return len(self.features)

    def __getitem__(self, idx):
        # 1行分のデータだけを返します
        return self.features[idx]