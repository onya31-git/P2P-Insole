# 可視化用コード
#
# 可視化UIもログの出力も、もっとリッチにしたいよね
#
#
import argparse
import pandas as pd
import io
import sys
import plotly.graph_objects as go
from processor.loader import load_config

def start(args):

    # YAML設定読み込み
    # config = load_config(args, args.config, args.model)

    # set real and pred skeleton data path
    file_path_real = "./data/test_data/Skeleton/T005S008_skeleton.csv"
    file_path_predict = "./output/predicted_skeleton.csv"

    # 同
    start_frame = 0
    # end_frame = min(len(frames_data_real), len(frames_data_pred))
    step = 50

    # skeleton definition
    bones = [
        (0, 1), (1, 2), (2, 3), (3, 4),                               # 脊椎             # 左右で分けて細かく改行する
        (5, 6), (6, 7), (7, 8), (9, 10), (10, 11), (11, 12), (5, 9),  # 手、肘、肩        # 左右で分けて細かく改行する
        (13, 14), (14, 15), (15, 16), (17, 18), (18, 19), (19, 20), (13, 17)  # 足、腰   # 左右で分けて細かく改行する
    ]

    # データの列を整える
    def process_skeleton_data(df):
        frames_data = []
        num_joints = len(df.columns) // 3

        for index, row in df.iterrows():
            x_positions, y_positions, z_positions = [], [], []

            for i in range(num_joints):
                try:
                    x = row[f'X.{i}']
                    y = row[f'Y.{i}']
                    z = row[f'Z.{i}']
                except KeyError:
                    continue

                if pd.notna(x) and pd.notna(y) and pd.notna(z):
                    x_positions.append(float(x))
                    y_positions.append(float(y))
                    z_positions.append(float(z))

            if x_positions:
                frames_data.append({'x': x_positions, 'y': y_positions, 'z': z_positions})
        return frames_data


    # make skeleton per frame (real: Red, pred: Blue)
    def create_frame_traces(frame_real, frame_pred):
        traces = []

        # real skeleton(Red)
        traces.append(go.Scatter3d(
            x=frame_real['x'],
            y=frame_real['z'],
            z=frame_real['y'],
            mode='markers',
            marker=dict(size=5, color='red', opacity=0.8),
            name='Real Joints'
        ))
        for start, end in bones:
            if start < len(frame_real['x']) and end < len(frame_real['x']):
                traces.append(go.Scatter3d(
                    x=[frame_real['x'][start], frame_real['x'][end]],
                    y=[frame_real['z'][start], frame_real['z'][end]],
                    z=[frame_real['y'][start], frame_real['y'][end]],
                    mode='lines',
                    line=dict(color='red', width=2),
                    name=f'Real Bone {start}-{end}'
                ))

        # pred skeleton(blue)
        traces.append(go.Scatter3d(
            x=frame_pred['x'],
            y=frame_pred['z'],
            z=frame_pred['y'],
            mode='markers',
            marker=dict(size=5, color='blue', opacity=0.8),
            name='Predicted Joints'
        ))
        for start, end in bones:
            if start < len(frame_pred['x']) and end < len(frame_pred['x']):
                traces.append(go.Scatter3d(
                    x=[frame_pred['x'][start], frame_pred['x'][end]],
                    y=[frame_pred['z'][start], frame_pred['z'][end]],
                    z=[frame_pred['y'][start], frame_pred['y'][end]],
                    mode='lines',
                    line=dict(color='blue', width=2),
                    name=f'Pred Bone {start}-{end}'
                ))

        return traces


    # main processing
    try:
        # load data
        df_real = pd.read_csv(file_path_real)
        df_pred = pd.read_csv(file_path_predict)
        print(f"Real data: {df_real.shape}")
        print(f"Pred data: {df_pred.shape}")
        frames_data_real = process_skeleton_data(df_real)
        frames_data_pred = process_skeleton_data(df_pred)

        end_frame = min(len(frames_data_real), len(frames_data_pred))

        frames_data_real = frames_data_real[start_frame:end_frame:step]
        frames_data_pred = frames_data_pred[start_frame:end_frame:step]

        # グラフィックオブジェクトの作成
        fig = go.Figure()

        # 初期フレーム
        initial_traces = create_frame_traces(frames_data_real[0], frames_data_pred[0])
        for trace in initial_traces:
            fig.add_trace(trace)

        # アニメーションフレームの作成
        frames = [
            go.Frame(
                data=create_frame_traces(real, pred),
                name=f'frame{i}'
            )
            for i, (real, pred) in enumerate(zip(frames_data_real, frames_data_pred))
        ]
        fig.frames = frames

        # レイアウトとコントローラ
        fig.update_layout(
            title='3D Skeleton Animation: Real (Red) vs Prediction (Blue)',
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

        # アニメーションを表示 & HTMLにエクスポート
        html_str = fig.to_html(full_html=True, include_plotlyjs='cdn')
        with open("./animation/animation.html", "w", encoding="utf-8") as f:
            f.write(html_str)

        print(f"num frames: {len(frames)}")
        print(f"Visualization is successful!")

    except Exception as e:
        print(f"An error has happened: {str(e)}")


@staticmethod
def get_parser(add_help=False):
    parser = argparse.ArgumentParser(add_help=add_help, description='Transformer Base Processor')

    parser.add_argument('-w', '--work_dir', default='', help='the work folder for storing results')
    parser.add_argument('-c', '--config', default=None, help='path to the configuration file')

    # Processor
    # feeder
    # model

    return parser