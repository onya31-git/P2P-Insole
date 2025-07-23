import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error

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

# ファイルパスを指定
truth_file = "./data/test_data/Skeleton/T005S008_skeleton.csv"
predicted_file = "./output/predicted_skeleton.csv"

import pandas as pdc
import numpy as np

def calculate_point_metrics(mocap_file, predicted_file):
    # CSVファイルの読み込み
    mocap_data = pd.read_csv(mocap_file, header=0)
    predicted_data = pd.read_csv(predicted_file, header=0)
    
    if mocap_data.shape != predicted_data.shape:
        raise ValueError("モーションキャプチャデータと予測データの形状が一致しません")
    
    num_points = mocap_data.shape[1] // 3
    point_metrics = {}
    
    # 全ポイントの統計量を格納するリスト
    all_mse_values = []
    all_rmse_values = []
    
    for point_idx in range(num_points):
        start_idx = point_idx * 3
        indices = [start_idx, start_idx + 1, start_idx + 2]
        
        mse_per_frame = []
        rmse_per_frame = []
        
        for frame_idx in range(len(mocap_data)):
            mocap_point = mocap_data.iloc[frame_idx][indices].values
            predicted_point = predicted_data.iloc[frame_idx][indices].values
            
            squared_diff = np.square(mocap_point - predicted_point)
            mse = np.mean(squared_diff)
            mse_per_frame.append(mse)
            
            rmse = np.sqrt(mse)
            rmse_per_frame.append(rmse)
        
        all_mse_values.extend(mse_per_frame)
        all_rmse_values.extend(rmse_per_frame)
        
        point_metrics[f'Point_{point_idx+1}'] = {
            'MSE': {
                'min': np.min(mse_per_frame),
                'max': np.max(mse_per_frame),
                'mean': np.mean(mse_per_frame),
                'median': np.median(mse_per_frame),
                'std': np.std(mse_per_frame)
            },
            'RMSE': {
                'min': np.min(rmse_per_frame),
                'max': np.max(rmse_per_frame),
                'mean': np.mean(rmse_per_frame),
                'median': np.median(rmse_per_frame),
                'std': np.std(rmse_per_frame)
            }
        }
    
    # 全ポイントの統計量を計算
    point_metrics['All_Points'] = {
        'MSE': {
            'min': np.min(all_mse_values),
            'max': np.max(all_mse_values),
            'mean': np.mean(all_mse_values),
            'median': np.median(all_mse_values),
            'std': np.std(all_mse_values)
        },
        'RMSE': {
            'min': np.min(all_rmse_values),
            'max': np.max(all_rmse_values),
            'mean': np.mean(all_rmse_values),
            'median': np.median(all_rmse_values),
            'std': np.std(all_rmse_values)
        }
    }
    
    return point_metrics

def print_metrics_table(metrics):
    points = list(metrics.keys())[:-1]  # All_Points を除外
    stats = ['min', 'max', 'mean', 'median', 'std']
    points_per_line = 11
    
    # MSEテーブルの表示
    print("\nMSE Statistics:")
    
    # 11ポイントずつ処理
    for i in range(0, len(points), points_per_line):
        current_points = points[i:i + points_per_line]
        
        print("-" * (10 + 15 * len(current_points)))
        print(f"{'Metric':<10}", end="")
        for point in current_points:
            print(f"{point:<15}", end="")
        print()
        print("-" * (10 + 15 * len(current_points)))
        
        for stat in stats:
            print(f"{stat:<10}", end="")
            for point in current_points:
                print(f"{metrics[point]['MSE'][stat]:15.6f}", end="")
            print()
        print()
    
    # 全ポイントの統計量を表示
    print("\nAll Points MSE Statistics:")
    print("-" * 25)
    for stat in stats:
        print(f"{stat:<10}: {metrics['All_Points']['MSE'][stat]:10.6f}")
    
    # RMSEテーブルの表示
    print("\nRMSE Statistics:")
    
    # 11ポイントずつ処理
    for i in range(0, len(points), points_per_line):
        current_points = points[i:i + points_per_line]
        
        print("-" * (10 + 15 * len(current_points)))
        print(f"{'Metric':<10}", end="")
        for point in current_points:
            print(f"{point:<15}", end="")
        print()
        print("-" * (10 + 15 * len(current_points)))
        
        for stat in stats:
            print(f"{stat:<10}", end="")
            for point in current_points:
                print(f"{metrics[point]['RMSE'][stat]:15.6f}", end="")
            print()
        print()
    
    # 全ポイントの統計量を表示
    print("\nAll Points RMSE Statistics:")
    print("-" * 25)
    for stat in stats:
        print(f"{stat:<10}: {metrics['All_Points']['RMSE'][stat]:10.6f}")

def main():
    
    try:
        metrics = calculate_point_metrics(truth_file, predicted_file)
        print_metrics_table(metrics)
            
    except Exception as e:
        print(f"エラーが発生しました: {str(e)}")

if __name__ == "__main__":
    main()