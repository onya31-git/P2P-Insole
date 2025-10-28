# main.py
import torch
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from transformer_model import EnhancedSkeletonTransformer, PressureSkeletonDataset, train_model, EnhancedSkeletonLoss

def preprocess_pressure_data(left_data, right_data):
    """圧力、回転、加速度データの前処理"""
    
    # 左足データから各種センサー値を抽出
    left_pressure = left_data.iloc[:, :35]  # 圧力センサーの列を適切に指定
    left_rotation = left_data.iloc[:, 35:38]  # 回転データの列を適切に指定
    left_accel = left_data.iloc[:, 38:41]  # 加速度データの列を適切に指定

    # 右足データから各種センサー値を抽出
    right_pressure = right_data.iloc[:, :35]  # 圧力センサーの列を適切に指定
    right_rotation = right_data.iloc[:, 35:38]  # 回転データの列を適切に指定
    right_accel = right_data.iloc[:, 38:41]  # 加速度データの列を適切に指定

    # データの結合
    pressure_combined = pd.concat([left_pressure, right_pressure], axis=1)
    rotation_combined = pd.concat([left_rotation, right_rotation], axis=1)
    accel_combined = pd.concat([left_accel, right_accel], axis=1)

    # NaN値を補正
    pressure_combined = pressure_combined.fillna(0.0)
    rotation_combined = rotation_combined.fillna(0.0)
    accel_combined = accel_combined.fillna(0.0)

    print("Checking pressure data for NaN or Inf...")
    print("Pressure NaN count:", pressure_combined.isna().sum().sum())
    print("Pressure Inf count:", np.isinf(pressure_combined).sum().sum())

    # 移動平均フィルタの適用
    window_size = 3
    pressure_combined = pressure_combined.rolling(window=window_size, center=True).mean()
    rotation_combined = rotation_combined.rolling(window=window_size, center=True).mean()
    accel_combined = accel_combined.rolling(window=window_size, center=True).mean()
    
    # NaN値を前後の値で補間
    pressure_combined = pressure_combined.fillna(method='bfill').fillna(method='ffill')
    rotation_combined = rotation_combined.fillna(method='bfill').fillna(method='ffill')
    accel_combined = accel_combined.fillna(method='bfill').fillna(method='ffill')

    # 正規化と標準化のスケーラー初期化
    pressure_normalizer = MinMaxScaler()
    rotation_normalizer = MinMaxScaler()
    accel_normalizer = MinMaxScaler()

    pressure_standardizer = StandardScaler(with_mean=True, with_std=True)
    rotation_standardizer = StandardScaler(with_mean=True, with_std=True)
    accel_standardizer = StandardScaler(with_mean=True, with_std=True)

    # データの正規化と標準化
    pressure_processed = pressure_standardizer.fit_transform(
        pressure_normalizer.fit_transform(pressure_combined)
    )
    rotation_processed = rotation_standardizer.fit_transform(
        rotation_normalizer.fit_transform(rotation_combined)
    )
    accel_processed = accel_standardizer.fit_transform(
        accel_normalizer.fit_transform(accel_combined)
    )

    # 1次微分と2次微分の計算
    pressure_grad1 = np.gradient(pressure_processed, axis=0)
    pressure_grad2 = np.gradient(pressure_grad1, axis=0)
    
    # 回転データと加速度データは積分を使うためコメントアウト(使用する場合は特徴量の結合を書き換える必要あり)
    rotation_grad1 = np.gradient(rotation_processed, axis=0)
    rotation_grad2 = np.gradient(rotation_grad1, axis=0)
    
    accel_grad1 = np.gradient(accel_processed, axis=0)
    accel_grad2 = np.gradient(accel_grad1, axis=0)

    # 一次積分と二次積分の計算(dt = 0.01(サンプリング間隔)は仮設定)
    # rotation_int1 = np.cumsum(rotation_processed * 0.01, axis=0)
    # rotation_int2 = np.cumsum(rotation_int1 * 0.01, axis=0)

    # accel_int1 = np.cumsum(accel_processed * 0.01, axis=0)
    # accel_int2 = np.cumsum(accel_int1 * 0.01, axis=0)


    # 特徴量の結合
    input_features = np.concatenate([
        pressure_processed,
        pressure_grad1,
        pressure_grad2,
        rotation_processed,
        rotation_grad1,
        rotation_grad2,
        accel_processed,
        accel_grad1,
        accel_grad2
    ], axis=1)

    return input_features, {
        'pressure': {
            'normalizer': pressure_normalizer,
            'standardizer': pressure_standardizer
        },
        'rotation': {
            'normalizer': rotation_normalizer,
            'standardizer': rotation_standardizer
        },
        'accel': {
            'normalizer': accel_normalizer,
            'standardizer': accel_standardizer
        }
    }

def load_and_combine_data(file_pairs):
    """複数のデータセットを読み込んで結合する"""
    all_skeleton_data = []
    all_pressure_left = []
    all_pressure_right = []
    
    for skeleton_file, left_file, right_file in file_pairs:
        skeleton = pd.read_csv(skeleton_file)
        left = pd.read_csv(left_file, dtype=float, low_memory=False)
        right = pd.read_csv(right_file, dtype=float, low_memory=False)

        # データ長を揃える
        min_length = min(len(skeleton), len(left), len(right))
        
        all_skeleton_data.append(skeleton.iloc[:min_length])
        all_pressure_left.append(left.iloc[:min_length])
        all_pressure_right.append(right.iloc[:min_length])
    
    return (pd.concat(all_skeleton_data, ignore_index=True),
            pd.concat(all_pressure_left, ignore_index=True),
            pd.concat(all_pressure_right, ignore_index=True))

def main():
    # データの読み込み
    data_pairs = [
        # #
        # # 第三回収集データ
        # #
        # # 立ちっぱなし
        # ('./data/20250517old_data/20241115test3/Opti-track/Take 2024-11-15 03.20.00 PM.csv',
        #  './data/20250517old_data/20241115test3/insoleSensor/20241115_152500_left.csv',
        #  './data/20250517old_data/20241115test3/insoleSensor/20241115_152500_right.csv'),
        # # お辞儀
        # ('./data/20250517old_data/20241115test3/Opti-track/Take 2024-11-15 03.26.00 PM.csv',
        #  './data/20250517old_data/20241115test3/insoleSensor/20241115_153100_left.csv', 
        #  './data/20250517old_data/20241115test3/insoleSensor/20241115_153100_right.csv'),
        # # 体の横の傾け
        # ('./data/20250517old_data/20241115test3/Opti-track/Take 2024-11-15 03.32.00 PM.csv', 
        #  './data/20250517old_data/20241115test3/insoleSensor/20241115_153700_left.csv', 
        #  './data/20250517old_data/20241115test3/insoleSensor/20241115_153700_right.csv'),
        # # 立つ座る
        # ('./data/20250517old_data/20241115test3/Opti-track/Take 2024-11-15 03.38.00 PM.csv', 
        #  './data/20250517old_data/20241115test3/insoleSensor/20241115_154300_left.csv', 
        #  './data/20250517old_data/20241115test3/insoleSensor/20241115_154300_right.csv'),
        # # スクワット
        # ('./data/20250517old_data/20241115test3/Opti-track/Take 2024-11-15 03.44.00 PM.csv', 
        #  './data/20250517old_data/20241115test3/insoleSensor/20241115_154900_left.csv', 
        #  './data/20250517old_data/20241115test3/insoleSensor/20241115_154900_right.csv'),
        #  # 総合(test3)
        # ('./data/20250517old_data/20241115test3/Opti-track/Take 2024-11-15 03.50.00 PM.csv', 
        #  './data/20250517old_data/20241115test3/insoleSensor/20241115_155500_left.csv', 
        #  './data/20250517old_data/20241115test3/insoleSensor/20241115_155500_right.csv'),

        # # 釘宮くん
        # ('./data/20250517old_data/20241212test4/Opti-track/Take 2024-12-12 03.06.59 PM.csv',
        #  './data/20250517old_data/20241212test4/insoleSensor/20241212_152700_left.csv', 
        #  './data/20250517old_data/20241212test4/insoleSensor/20241212_152700_right.csv'),
        # # 百田くん
        # ('./data/20250517old_data/20241212test4/Opti-track/Take 2024-12-12 03.45.00 PM.csv', 
        #  './data/20250517old_data/20241212test4/insoleSensor/20241212_160501_left.csv', 
        #  './data/20250517old_data/20241212test4/insoleSensor/20241212_160501_right.csv'),
        # # # # 渡辺(me)
        # ('./data/20250517old_data/20241212test4/Opti-track/Take 2024-12-12 04.28.00 PM.csv', 
        #  './data/20250517old_data/20241212test4/insoleSensor/20241212_164800_left.csv', 
        #  './data/20250517old_data/20241212test4/insoleSensor/20241212_164800_right.csv'),
        # # にるぱむさん
        # ('./data/20250517old_data/20241212test4/Opti-track/Take 2024-12-12 05.17.59 PM.csv', 
        #  './data/20250517old_data/20241212test4/insoleSensor/20241212_173800_left.csv', 
        #  './data/20250517old_data/20241212test4/insoleSensor/20241212_173800_right.csv')


         # 新データ(test5)
         # s1
        ('./data/training_data/Skeleton/T005S001_skeleton.csv', 
         './data/training_data/Insole/T005S001_Insole_l.csv', 
         './data/training_data/Insole/T005S001_Insole_r.csv'),
        # s2
        ('./data/training_data/Skeleton/T005S002_skeleton.csv', 
         './data/training_data/Insole/T005S002_Insole_l.csv', 
         './data/training_data/Insole/T005S002_Insole_r.csv'),
        # s3
        ('./data/training_data/Skeleton/T005S003_skeleton.csv', 
         './data/training_data/Insole/T005S003_Insole_l.csv', 
         './data/training_data/Insole/T005S003_Insole_r.csv'),
        # s4
        ('./data/training_data/Skeleton/T005S004_skeleton.csv', 
         './data/training_data/Insole/T005S004_Insole_l.csv', 
         './data/training_data/Insole/T005S004_Insole_r.csv'),
        # s5
        ('./data/training_data/Skeleton/T005S005_skeleton.csv', 
         './data/training_data/Insole/T005S005_Insole_l.csv', 
         './data/training_data/Insole/T005S005_Insole_r.csv'),
         # s6
        ('./data/training_data/Skeleton/T005S006_skeleton.csv', 
         './data/training_data/Insole/T005S006_Insole_l.csv', 
         './data/training_data/Insole/T005S006_Insole_r.csv'),
         # s7
        ('./data/training_data/Skeleton/T005S007_skeleton.csv', 
         './data/training_data/Insole/T005S007_Insole_l.csv', 
         './data/training_data/Insole/T005S007_Insole_r.csv'),
    ]
    
    # データの読み込みと結合
    skeleton_data, pressure_data_left, pressure_data_right = load_and_combine_data(data_pairs)

    # numpy配列に変換
    skeleton_data = skeleton_data.to_numpy()

    # 圧力、回転、加速度データの前処理
    input_features, sensor_scalers = preprocess_pressure_data(
        pressure_data_left,
        pressure_data_right
    )

    # データの分割
    train_input, val_input, train_skeleton, val_skeleton = train_test_split(
        input_features, 
        skeleton_data,
        test_size=0.2, 
        random_state=42
    )

    # デバイスの設定
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # モデルのパラメータ設定
    input_dim = input_features.shape[1]  # 圧力+回転+加速度の合計次元数
    d_model = 512
    nhead = 8
    num_encoder_layers = 8
    num_joints = 21 # skeleton_data.shape[1] // 3  # 3D座標なので3で割る
    dropout = 0.2
    batch_size = 32

    # データローダーの設定
    train_dataset = PressureSkeletonDataset(train_input, train_skeleton)
    val_dataset = PressureSkeletonDataset(val_input, val_skeleton)
    
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

    print("Checking final training and validation data...")
    print("Train input NaN count:", np.isnan(train_input).sum(), "Inf count:", np.isinf(train_input).sum())
    print("Train skeleton NaN count:", np.isnan(train_skeleton).sum(), "Inf count:", np.isinf(train_skeleton).sum())


    # モデルの初期化
    model = EnhancedSkeletonTransformer(
        input_dim= input_features.shape[1], # input_dim,
        d_model= d_model,
        nhead= nhead,
        num_encoder_layers= num_encoder_layers,
        num_joints=num_joints,
        num_dims=3,
        dropout=dropout
    ).to(device)

    # 損失関数、オプティマイザ、スケジューラの設定
    # criterion = torch.nn.MSELoss()  # 必要に応じてカスタム損失関数に変更可能
    criterion = EnhancedSkeletonLoss(alpha=1.0, beta=0.1)
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
        # verbose=True
    )

    # トレーニング実行
    train_model(
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

    # モデルの保存
    final_checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'sensor_scalers': sensor_scalers,
        # 'skeleton_skaler': skeleton_scaler,
        'model_config': {
            'input_dim': input_dim,
            'd_model': d_model,
            'nhead': nhead,
            'num_encoder_layers': num_encoder_layers,
            'num_joints': num_joints
        }
    }
    torch.save(final_checkpoint, './weight/final_skeleton_model.pth')

if __name__ == "__main__":
    main()