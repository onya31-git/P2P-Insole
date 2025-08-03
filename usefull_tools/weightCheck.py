import torch

weight_path = "./weight/best_skeleton_model.pth"  # 重みファイル名
checkpoint = torch.load(weight_path, map_location="cpu")  # GPUを使わないならcpu


for name, param in checkpoint["model_state_dict"].items():
    print(name, param.shape)

for name, param in checkpoint["model_state_dict"].items():
    print(name, param.mean(), param.std())
