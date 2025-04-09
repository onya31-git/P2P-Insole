import pandas as pd
import plotly.graph_objects as go
import numpy as np

# ファイルパスの設定
file_path = "./data/20241115test3/Opti-track/Take 2024-11-15 03.32.00 PM.csv"  # パスを適切に設定

# file_path = "./output/predicted_skeleton.csv"

# ボーンの接続関係を定義
bones = [
    (0, 1), (1, 2), (2, 3), (3, 4),  # 背骨
    (5, 6), (6, 7), (7, 8), (9, 10), (10, 11), (11, 12),  (5, 9), # 腕と肩
    (13, 14), (14, 15), (15, 16), (17, 18), (18, 19), (19, 20), (13, 17), # 足と腰
]

# データの読み込みと処理を修正
def process_skeleton_data(df):
    frames_data = []
    
    # 3つずつの列をグループ化してx, y, z座標として処理
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
    
    return frames_data

# データの読み込みと処理
try:
    df = pd.read_csv(file_path)
    print("Data loaded successfully")
    print(f"DataFrame shape: {df.shape}")
    
    frames_data = process_skeleton_data(df)
    print(f"Processed {len(frames_data)} frames")
    
    if not frames_data:
        raise ValueError("No valid frames were processed")
        
    # 表示するフレームの範囲を設定
    start_frame = 0
    end_frame = len(frames_data)
    step = 5  # フレームの間隔
    frames_data = frames_data[start_frame:end_frame:step]
    
    fig = go.Figure()

    # 初期フレームの作成（マーカーと線の両方を含む）
    def create_frame_traces(frame_data):
        traces = []
        
        # マーカー（関節点）の追加
        traces.append(
            go.Scatter3d(
                x=frame_data['x'],
                y=frame_data['z'],
                z=frame_data['y'],
                mode='markers',
                marker=dict(size=5, color='blue', opacity=0.8),
                name='Joints'
            )
        )
        
        # ボーン（線）の追加
        for start, end in bones:
            if start < len(frame_data['x']) and end < len(frame_data['x']):
                traces.append(
                    go.Scatter3d(
                        x=[frame_data['x'][start], frame_data['x'][end]],
                        y=[frame_data['z'][start], frame_data['z'][end]],
                        z=[frame_data['y'][start], frame_data['y'][end]],
                        mode='lines',
                        line=dict(color='red', width=2),
                        name=f'Bone {start}-{end}'
                    )
                )
        
        return traces

    # 初期フレームの追加
    initial_traces = create_frame_traces(frames_data[0])
    for trace in initial_traces:
        fig.add_trace(trace)

    # フレームの作成
    frames = [
        go.Frame(
            data=create_frame_traces(frame),
            name=f'frame{i}'
        )
        for i, frame in enumerate(frames_data)
    ]

    fig.frames = frames

    # レイアウトの設定
    fig.update_layout(
        title='Skeleton Data Animation',
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z',
            aspectmode='data'
        ),
        width=1000,
        height=1000,
        showlegend=True,
        updatemenus=[
            {
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
            }
        ],
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

    fig.show()
    fig.write_html("./animation/animation.html")

    print(f"Total number of frames: {len(frames_data)}")

except Exception as e:
    print(f"An error occurred: {str(e)}")