import pandas as pd
import numpy as np

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

def main():
    truth_file = "./data/20241115test3/Opti-track/Take 2024-11-15 03.44.00 PM.csv"
    predicted_file = "./output/skeleton11_test4_squat.csv"
    
    try:
        metrics = calculate_point_metrics(truth_file, predicted_file)
        print_metrics_table(metrics)
            
    except Exception as e:
        print(f"エラーが発生しました: {str(e)}")

def calculate_point_metrics(mocap_file, predicted_file):
    mocap_data = pd.read_csv(mocap_file, header=0)
    predicted_data = pd.read_csv(predicted_file, header=0)
    
    if mocap_data.shape != predicted_data.shape:
        raise ValueError("モーションキャプチャデータと予測データの形状が一致しません")
    
    num_points = mocap_data.shape[1] // 3
    point_metrics = {}
    
    # ポイントグループの定義
    point_groups = {
        'Spine': [0, 1, 2],
        'Head': [3],
        'Arms': [5, 6, 7, 9, 10, 11],
        'Legs': [12, 13, 14, 15, 16, 17, 18, 19, 20]
    }
    
    for group_name, point_indices in point_groups.items():
        group_mse_values = []
        group_rmse_values = []
        
        for point_idx in point_indices:
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
            
            group_mse_values.extend(mse_per_frame)
            group_rmse_values.extend(rmse_per_frame)
        
        point_metrics[group_name] = {
            'MSE': {
                'min': np.min(group_mse_values),
                'max': np.max(group_mse_values),
                'mean': np.mean(group_mse_values),
                'median': np.median(group_mse_values),
                'std': np.std(group_mse_values)
            },
            'RMSE': {
                'min': np.min(group_rmse_values),
                'max': np.max(group_rmse_values),
                'mean': np.mean(group_rmse_values),
                'median': np.median(group_rmse_values),
                'std': np.std(group_rmse_values)
            }
        }
    
    return point_metrics

def print_metrics_table(metrics):
    stats = ['min', 'max', 'mean', 'median', 'std']
    
    for group_name, group_metrics in metrics.items():
        print(f"\n{group_name} Metrics:")
        print("-" * 25)
        for stat in stats:
            print(f"{stat:<10}: MSE={group_metrics['MSE'][stat]:10.6f}, RMSE={group_metrics['RMSE'][stat]:10.6f}")

if __name__ == "__main__":
    main()