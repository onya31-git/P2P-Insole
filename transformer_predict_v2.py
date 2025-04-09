import torch
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from transformer_model import EnhancedSkeletonTransformer 

def preprocess_pressure_data(left_data, right_data):
    """圧力、回転、加速度データの前処理"""
    # 左足データから各種センサー値を抽出
    left_pressure = left_data.iloc[:, :35]
    left_rotation = left_data.iloc[:, 35:38]
    left_accel = left_data.iloc[:, 38:41]

    # 右足データから各種センサー値を抽出
    right_pressure = right_data.iloc[:, :35]
    right_rotation = right_data.iloc[:, 35:38]
    right_accel = right_data.iloc[:, 38:41]

    # データの結合
    pressure_combined = pd.concat([left_pressure, right_pressure], axis=1)
    rotation_combined = pd.concat([left_rotation, right_rotation], axis=1)
    accel_combined = pd.concat([left_accel, right_accel], axis=1)

    # NaN値を補正
    pressure_combined = pressure_combined.ffill().bfill()
    rotation_combined = rotation_combined.ffill().bfill()
    accel_combined = accel_combined.ffill().bfill()

    # 移動平均フィルタの適用
    window_size = 3
    pressure_combined = pressure_combined.rolling(window=window_size, center=True).mean()
    rotation_combined = rotation_combined.rolling(window=window_size, center=True).mean()
    accel_combined = accel_combined.rolling(window=window_size, center=True).mean()
    
    # NaN値を補間
    pressure_combined = pressure_combined.ffill().bfill()
    rotation_combined = rotation_combined.ffill().bfill()
    accel_combined = accel_combined.ffill().bfill()

    # 正規化と標準化
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
    
    # 加速度と回転には積分を適応する方針なのでコメントアウト(使用する場合は特徴量の結合を書き換える)
    rotation_grad1 = np.gradient(rotation_processed, axis=0)
    rotation_grad2 = np.gradient(rotation_grad1, axis=0)
    
    accel_grad1 = np.gradient(accel_processed, axis=0)
    accel_grad2 = np.gradient(accel_grad1, axis=0)

    # 一次積分と二次積分の計算(dt = 0.01(サンプリング間隔)は仮設定)
    # rotation_int1 = np.cumsum(rotation_processed * 0.01, axis=0)
    # rotation_int2 = np.cumsum(rotation_int1 * 0.01, axis=0)

    # accel_int1 = np.cumsum(accel_processed * 0.01, axis=0)
    # accel_int2 = np.cumsum(accel_int1 * 0.01, axis=0)

    # 特徴量の結合（246次元になるはず）
    input_features = np.concatenate([
        pressure_processed,  # 原特徴量
        pressure_grad1,     # 1次微分
        pressure_grad2,     # 2次微分
        rotation_processed,
        rotation_grad1,
        rotation_grad2,
        accel_processed,
        accel_grad1,
        accel_grad2
    ], axis=1)

    return input_features

def load_and_preprocess_data(file_pairs):
    predictions_all = []
    
    for skeleton_file, left_file, right_file in file_pairs:
        skeleton_data = pd.read_csv(skeleton_file)
        pressure_data_left = pd.read_csv(left_file)
        pressure_data_right = pd.read_csv(right_file)
        
        input_features = preprocess_pressure_data(pressure_data_left, pressure_data_right)
        min_length = min(len(skeleton_data), len(input_features))
        
        input_features = input_features.iloc[:min_length]
        skeleton_data = skeleton_data.iloc[:min_length]
        
        predictions_all.append((input_features, skeleton_data))
    
    return predictions_all

def predict_skeleton():
        # 立ちっぱなし
        # ('./data/20241115test3/Opti-track/Take 2024-11-15 03.20.00 PM.csv',
        #  './data/20241115test3/insoleSensor/20241115_152500_left.csv',
        #  './data/20241115test3/insoleSensor/20241115_152500_right.csv'),
        # お辞儀
        # ('./data/20241115test3/Opti-track/Take 2024-11-15 03.26.00 PM.csv',
        #  './data/20241115test3/insoleSensor/20241115_153100_left.csv', 
        #  './data/20241115test3/insoleSensor/20241115_153100_right.csv'),
        # # 体の横の傾け
        # ('./data/20241115test3/Opti-track/Take 2024-11-15 03.32.00 PM.csv', 
        #  './data/20241115test3/insoleSensor/20241115_153700_left.csv', 
        #  './data/20241115test3/insoleSensor/20241115_153700_right.csv'),
        # # 立つ座る
        # ('./data/20241115test3/Opti-track/Take 2024-11-15 03.38.00 PM.csv', 
        #  './data/20241115test3/insoleSensor/20241115_154300_left.csv', 
        #  './data/20241115test3/insoleSensor/20241115_154300_right.csv'),
        # # スクワット
        # ('./data/20241115test3/Opti-track/Take 2024-11-15 03.44.00 PM.csv', 
        #  './data/20241115test3/insoleSensor/20241115_154900_left.csv', 
        #  './data/20241115test3/insoleSensor/20241115_154900_right.csv'),

    try:
        # データの読み込みと前処理
        skeleton_data = pd.read_csv('./data/20241115test3/Opti-track/Take 2024-11-15 03.38.00 PM.csv')
        pressure_data_left = pd.read_csv('./data/20241115test3/insoleSensor/20241115_154300_left.csv', skiprows=1)
        pressure_data_right = pd.read_csv('./data/20241115test3/insoleSensor/20241115_154300_right.csv', skiprows=1)

        # 入力データの前処理
        input_features = preprocess_pressure_data(pressure_data_left, pressure_data_right)
        
        # 入力の次元数を取得
        input_dim = input_features.shape[1]
        num_joints = skeleton_data.shape[1] // 3

        # デバイスの設定
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")

        # モデルの初期化（固定パラメータを使用）
        model = EnhancedSkeletonTransformer(
            input_dim=input_dim,
            d_model=512,
            nhead=8,
            num_encoder_layers=8,
            num_joints=num_joints,
            num_dims=3,
            dropout=0.2
        ).to(device)

        # チェックポイントの読み込み（weights_only=Trueを追加）
        checkpoint = torch.load('./weight/best_skeleton_model_test4Tasks.pth', map_location=device, weights_only=True)
        
        # モデルの重みを読み込み
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        print("Model loaded successfully")

        # 予測の実行
        print("Making predictions...")
        with torch.no_grad():
            input_tensor = torch.FloatTensor(input_features).to(device)
            predictions = model(input_tensor)
            predictions = predictions.cpu().numpy()

        print(f"Prediction shape: {predictions.shape}")
        return predictions

    except Exception as e:
        print(f"Error during prediction: {str(e)}")
        raise

def save_predictions(predictions, output_file='./output/predicted_skeleton.csv'):
    try:
        # 予測結果をデータフレームに変換
        num_joints = predictions.shape[1] // 3
        columns = []
        for i in range(num_joints):
            columns.extend([f'joint_{i}_x', f'joint_{i}_y', f'joint_{i}_z'])
        
        df_predictions = pd.DataFrame(predictions, columns=columns)
        df_predictions.to_csv(output_file, index=False)
        print(f"Predictions saved to {output_file}")
        
    except Exception as e:
        print(f"Error saving predictions: {str(e)}")
        raise

def main():
    try:
        print("Starting prediction process...")
        predictions = predict_skeleton()
        
        print("\nSaving predictions...")
        save_predictions(predictions)
        print(predictions)
        
        print("Prediction process completed successfully!")
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()