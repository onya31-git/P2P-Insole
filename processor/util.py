# カテゴリのわからないファイルを集約
#
#
#
#

def print_config(params):
    print("<< Final Configuration Settings >>", flush=True)
    print("[Model Parameters]")
    print(f"  d_model: {params["d_model"]}")
    print(f"  n_head: {params["n_head"]}")
    print(f"  num_encoder_layer: {params["num_encoder_layer"]}")
    print(f"  dropout: {params["dropout"]}")
    print(f"  sequence_size: {params["sequence_len"]}")
    print(f"  use_gradient_data: {params["use_gradient_data"]}")
    print("\n[Training Parameters]")
    print(f"  Epochs: {params["num_epoch"]}")
    print(f"  Batch Size: {params["batch_size"]}")
    print("\n[Optimization Parameters]")
    print(f"  Learning Rate: {params["learning_rate"]}")
    print(f"  Weight Decay: {params["weight_decay"]}")
    print("\n[Loss Function Parameters]")
    print(f"  Loss Alpha: {params["loss_alpha"]}")
    print(f"  Loss Beta: {params["loss_beta"]}")
    print("\n[Other Settings]")
    print(f"  Input Dimension: {params["input_dim"]}")
    print(f"  Number of Joints: {params["num_joints"]}")
    print(f"  Number of Dimensions: {params["num_dims"]}")
    print("---" * 20)