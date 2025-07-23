import pandas as pd
import plotly.graph_objects as go
import plotly.offline as pyo

# === ファイル読み込み ===
file_path = "./output/predicted_skeleton.csv"  # ここを適宜書き換えてください
df = pd.read_csv(file_path)

# === 可視化パラメータの指定 ===
start_frame = 0          # 開始フレーム
end_frame = 5000# len(df) - 1  # 終了フレーム（デフォルトで最後のフレームまで）
frame_interval = 50       # 何フレームおきに可視化するか

# === パラメータ設定 ===
num_joints = 21
joint_labels = [f"{axis}.{i}" for i in range(num_joints) for axis in ['X', 'Y', 'Z']]
df.columns = joint_labels

# === アニメーションフレーム作成 ===
frames = []
for i in range(start_frame, end_frame + 1, frame_interval):
    xs = [df.loc[i, f"X.{j}"] for j in range(num_joints)]
    ys = [df.loc[i, f"Y.{j}"] for j in range(num_joints)]
    zs = [df.loc[i, f"Z.{j}"] for j in range(num_joints)]

    scatter = go.Scatter3d(
        x=xs, y=ys, z=zs,
        mode='markers+lines',
        marker=dict(size=4, color='blue'),
        line=dict(color='gray', width=2),
        name=f"Frame {i}"
    )
    frames.append(go.Frame(data=[scatter], name=str(i)))

# === 初期フレームのデータ ===
initial_frame = start_frame
initial_x = [df.loc[initial_frame, f"X.{j}"] for j in range(num_joints)]
initial_y = [df.loc[initial_frame, f"Y.{j}"] for j in range(num_joints)]
initial_z = [df.loc[initial_frame, f"Z.{j}"] for j in range(num_joints)]

# === Figure定義 ===
fig = go.Figure(
    data=[go.Scatter3d(x=initial_x, y=initial_y, z=initial_z, mode='markers+lines',
                       marker=dict(size=4, color='blue'),
                       line=dict(color='gray', width=2))],
    layout=go.Layout(
        title="3D Skeleton Animation",
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z',
            aspectmode='data'
        ),
        updatemenus=[dict(
            type="buttons",
            buttons=[dict(label="Play", method="animate", args=[None])]
        )]
    ),
    frames=frames
)

# === HTMLファイルとして保存 ===
pyo.plot(fig, filename="./animation/skeleton_animation.html", auto_open=False)
print("保存が完了しました: skeleton_animation.html")