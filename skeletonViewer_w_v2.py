import pandas as pd
import plotly.graph_objects as go
import numpy as np
import os

# ファイルパスの定義
file_path1 = "./data/20241115test3/Opti-track/Take 2024-11-15 03.32.00 PM.csv"
file_path2 = "./output/skeleton11_test4_yoko.csv"

# file_path1 = './data/20241115test3/Opti-track/Take 2024-11-15 03.38.00 PM.csv'
# file_path2 = "./output/predicted_skeleton.csv"

# ファイルの存在確認
for path in [file_path1, file_path2]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")

bones = [
    (0, 1), (1, 2), (2, 3), (3, 4),  # 背骨
    (5, 6), (6, 7), (7, 8), (9, 10), (10, 11), (11, 12),  (5, 9), # 腕と肩
    (13, 14), (14, 15), (15, 16), (17, 18), (18, 19), (19, 20), (13, 17), # 足と腰
]

def process_ground_truth_data(file_path):
    try:
        df = pd.read_csv(file_path, skiprows=6)
        print(f"Ground truth data loaded successfully. Shape: {df.shape}")
        frames_data = []

        for index, row in list(df.iterrows())[1: 10000: 5]:
            x_positions = []
            y_positions = []
            z_positions = []
            
            current_position = []
            for value in row:
                if pd.notna(value):
                    try:
                        current_position.append(float(value))
                        if len(current_position) == 3:
                            x_positions.append(current_position[0])
                            y_positions.append(current_position[1])
                            z_positions.append(current_position[2])
                            current_position = []
                    except (ValueError, TypeError):
                        continue
            
            if x_positions:
                frames_data.append({
                    'x': x_positions,
                    'y': y_positions,
                    'z': z_positions
                })
        
        print(f"Processed {len(frames_data)} ground truth frames")
        return frames_data
    except Exception as e:
        print(f"Error processing ground truth data: {str(e)}")
        raise

def process_predicted_data(file_path):
    try:
        df = pd.read_csv(file_path)
        print(f"Predicted data loaded successfully. Shape: {df.shape}")
        frames_data = []
        
        num_joints = len(df.columns) // 3
        
        for index, row in df.iterrows():
            x_positions = []
            y_positions = []
            z_positions = []
            
            for i in range(num_joints):
                x = row[f'joint_{i}_x']
                y = row[f'joint_{i}_y']
                z = row[f'joint_{i}_z']
                
                if pd.notna(x) and pd.notna(y) and pd.notna(z):
                    x_positions.append(float(x))
                    y_positions.append(float(y))
                    z_positions.append(float(z))
            
            if x_positions:
                frames_data.append({
                    'x': x_positions,
                    'y': y_positions,
                    'z': z_positions
                })
        
        print(f"Processed {len(frames_data)} predicted frames")
        return frames_data
    except Exception as e:
        print(f"Error processing predicted data: {str(e)}")
        raise

try:
    # データの読み込みと処理
    frames_data1 = process_ground_truth_data(file_path1)
    frames_data2 = process_predicted_data(file_path2)

    if not frames_data1 or not frames_data2:
        raise ValueError("No valid frames were processed")

    # フレーム数を揃える
    min_frames = min(len(frames_data1), len(frames_data2))
    frames_data1 = frames_data1[:min_frames]
    frames_data2 = frames_data2[:min_frames]
    
    print(f"Using {min_frames} frames for comparison")

    def create_frame_traces(frame_data1, frame_data2):
        traces = []
        
        # Ground Truth スケルトン（黄緑色）
        traces.append(
            go.Scatter3d(
                x=frame_data1['x'],
                y=frame_data1['z'],
                z=frame_data1['y'],
                mode='markers',
                marker=dict(size=5, color='lime', opacity=0.8),
                name='True Joints'
            )
        )
        
        for start, end in bones:
            if start < len(frame_data1['x']) and end < len(frame_data1['x']):
                traces.append(
                    go.Scatter3d(
                        x=[frame_data1['x'][start], frame_data1['x'][end]],
                        y=[frame_data1['z'][start], frame_data1['z'][end]],
                        z=[frame_data1['y'][start], frame_data1['y'][end]],
                        mode='lines',
                        line=dict(color='lime', width=2),
                        name='True Bones'
                    )
                )
        
        # 予測スケルトン（赤色）
        traces.append(
            go.Scatter3d(
                x=frame_data2['x'],
                y=frame_data2['z'],
                z=frame_data2['y'],
                mode='markers',
                marker=dict(size=5, color='red', opacity=0.8),
                name='Predict Joints'
            )
        )
        
        for start, end in bones:
            if start < len(frame_data2['x']) and end < len(frame_data2['x']):
                traces.append(
                    go.Scatter3d(
                        x=[frame_data2['x'][start], frame_data2['x'][end]],
                        y=[frame_data2['z'][start], frame_data2['z'][end]],
                        z=[frame_data2['y'][start], frame_data2['y'][end]],
                        mode='lines',
                        line=dict(color='red', width=2),
                        name='Predict Bones'
                    )
                )
        
        return traces

    fig = go.Figure()

    # 初期フレームの追加
    initial_traces = create_frame_traces(frames_data1[0], frames_data2[0])
    for trace in initial_traces:
        fig.add_trace(trace)

    # フレームの作成
    frames = [
        go.Frame(
            data=create_frame_traces(frame1, frame2),
            name=f'frame{i}'
        )
        for i, (frame1, frame2) in enumerate(zip(frames_data1, frames_data2))
    ]

    fig.frames = frames

    # レイアウトの設定
    fig.update_layout(
        title='Dual Skeleton Animation Comparison (Ground Truth vs Predicted)',
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z',
            aspectmode='data'
        ),
        width=1000,
        height=1000,
        showlegend=True,
        updatemenus=[{
            "buttons": [
                {
                    "args": [None, {"frame": {"duration": 50, "redraw": True},
                                  "fromcurrent": True}],
                    "label": "Play",
                    "method": "animate"
                },
                {
                    "args": [[None], {"frame": {"duration": 0, "redraw": True},
                                    "mode": "immediate",
                                    "transition": {"duration": 0}}],
                    "label": "Pause",
                    "method": "animate"
                }
            ],
            "direction": "left",
            "pad": {"r": 10, "t": 87},
            "showactive": False,
            "type": "buttons",
            "x": 0.1,
            "xanchor": "right",
            "y": 0,
            "yanchor": "top"
        }],
        sliders=[{
            "active": 0,
            "yanchor": "top",
            "xanchor": "left",
            "currentvalue": {
                "font": {"size": 20},
                "prefix": "Frame:",
                "visible": True,
                "xanchor": "right"
            },
            "transition": {"duration": 300, "easing": "cubic-in-out"},
            "pad": {"b": 10, "t": 50},
            "len": 0.9,
            "x": 0.1,
            "y": 0,
            "steps": [
                {
                    "args": [[f.name], {
                        "frame": {"duration": 0, "redraw": True},
                        "mode": "immediate",
                        "transition": {"duration": 0}
                    }],
                    "label": str(k),
                    "method": "animate"
                }
                for k, f in enumerate(frames)
            ]
        }]
    )

    # アニメーションの保存と表示
    fig.show()
    
    # アニメーション保存用ディレクトリの確認と作成
    os.makedirs("./animation", exist_ok=True)
    fig.write_html("./animation/dual_animation.html")

    print(f"Animation saved with {min_frames} frames")

except Exception as e:
    print(f"An error occurred: {str(e)}")