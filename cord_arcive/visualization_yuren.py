import pandas as pd
import plotly.graph_objects as go
import numpy as np

# 文件路径设置（请根据你的实际路径进行修改）
# file_path_real = "./data/test_data/Skeleton/T005S008_skeleton.csv"
# # file_path_real = "./data/20250518test3/Opti-track/3_final/Take 2024-11-15 03.49.59 PM.csv"
# file_path_predict = "./output/predicted_skeleton.csv"

file_path_real = "./data/20250517old_data/20241115test3/Opti-track/Take 2024-11-15 03.50.00 PM.csv"
file_path_predict = "./output/2DCNNTrans_predicted_skeleton.csv"

# 关节连接关系（骨架结构）
bones = [
    (0, 1), (1, 2), (2, 3), (3, 4),               # 脊柱
    (5, 6), (6, 7), (7, 8), (9, 10), (10, 11), (11, 12), (5, 9),  # 手臂与肩膀
    (13, 14), (14, 15), (15, 16), (17, 18), (18, 19), (19, 20), (13, 17)  # 腿部与腰
]

# 数据处理函数（每帧解析为字典：x, y, z）
def process_skeleton_data(df):
    frames_data = []
    num_joints = len(df.columns) // 3

    for index, row in df.iterrows():
        x_positions, y_positions, z_positions = [], [], []

        for i in range(num_joints):
            try:
                x = row[f'X.{2*i +1}']
                y = row[f'Y.{2*i +1}']
                z = row[f'Z.{2*i +1}']
            except KeyError:
                continue

            if pd.notna(x) and pd.notna(y) and pd.notna(z):
                x_positions.append(float(x))
                y_positions.append(float(y))
                z_positions.append(float(z))

        if x_positions:
            frames_data.append({'x': x_positions, 'y': y_positions, 'z': z_positions})
    return frames_data


# 创建每一帧的绘图元素（真实为红色，预测为黄色）
def create_frame_traces(frame_real, frame_pred):
    traces = []

    # 真实骨架（红色）
    traces.append(go.Scatter3d(
        x=frame_real['x'],
        y=frame_real['z'],
        z=frame_real['y'],
        mode='markers',
        marker=dict(size=5, color='lime', opacity=0.8),
        name='Real Joints'
    ))
    for start, end in bones:
        if start < len(frame_real['x']) and end < len(frame_real['x']):
            traces.append(go.Scatter3d(
                x=[frame_real['x'][start], frame_real['x'][end]],
                y=[frame_real['z'][start], frame_real['z'][end]],
                z=[frame_real['y'][start], frame_real['y'][end]],
                mode='lines',
                line=dict(color='lime', width=2),
                name=f'Real Bone {start}-{end}'
            ))

    # 预测骨架（黄色）
    traces.append(go.Scatter3d(
        x=frame_pred['x'],
        y=frame_pred['z'],
        z=frame_pred['y'],
        mode='markers',
        marker=dict(size=5, color='red', opacity=0.8),
        name='Predicted Joints'
    ))
    for start, end in bones:
        if start < len(frame_pred['x']) and end < len(frame_pred['x']):
            traces.append(go.Scatter3d(
                x=[frame_pred['x'][start], frame_pred['x'][end]],
                y=[frame_pred['z'][start], frame_pred['z'][end]],
                z=[frame_pred['y'][start], frame_pred['y'][end]],
                mode='lines',
                line=dict(color='red', width=2),
                name=f'Pred Bone {start}-{end}'
            ))

    return traces


# 主流程
try:
    # 加载数据
    df_real = pd.read_csv(file_path_real)
    df_pred = pd.read_csv(file_path_predict)
    print(f"Real data: {df_real.shape}")
    print(f"Pred data: {df_pred.shape}")

    # 处理数据
    frames_data_real = process_skeleton_data(df_real)
    frames_data_pred = process_skeleton_data(df_pred)

    # 同步帧数量和下采样
    start_frame = 0
    end_frame = min(len(frames_data_real), len(frames_data_pred))
    step = 5

    frames_data_real = frames_data_real[start_frame:end_frame:step]
    frames_data_pred = frames_data_pred[start_frame:end_frame:step]

    if not frames_data_real or not frames_data_pred:
        raise ValueError("未能成功处理帧数据。请检查CSV文件格式。")

    # 创建图形对象
    fig = go.Figure()

    # 初始帧
    initial_traces = create_frame_traces(frames_data_real[0], frames_data_pred[0])
    for trace in initial_traces:
        fig.add_trace(trace)

    # 创建动画帧
    frames = [
        go.Frame(
            data=create_frame_traces(real, pred),
            name=f'frame{i}'
        )
        for i, (real, pred) in enumerate(zip(frames_data_real, frames_data_pred))
    ]
    fig.frames = frames

    # 布局与控制器
    fig.update_layout(
        title='3D Skeleton Animation: Real (Red) vs Prediction (Green)',
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

    # 显示动画并导出为HTML
    fig.show()
    fig.write_html("./animation/animation.html")

    print(f"处理并可视化帧数: {len(frames)}")

except Exception as e:
    print(f"发生错误: {str(e)}")
