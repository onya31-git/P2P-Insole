# カテゴリのわからないファイルを集約
#
#
#
#

def print_config(config, input_dim, num_joints, num_dims):
    print("<< Final Configuration Settings >>", flush=True)
    print("[Model Parameters]")
    print(f"  d_model: {config["train"]["d_model"]}")
    print(f"  n_head: {config["train"]["n_head"]}")
    print(f"  num_encoder_layer: {config["train"]["num_encoder_layer"]}")
    print(f"  dropout: {config["train"]["dropout"]}")
    print(f"  sequence_size: {config["train"]["sequence_size"]}")
    print(f"  use_gradient_data: {config["train"]["use_gradient_data"]}")
    print("\n[Training Parameters]")
    print(f"  Epochs: {config["train"]["epoch"]}")
    print(f"  Batch Size: {config["train"]["batch_size"]}")
    print("\n[Optimization Parameters]")
    print(f"  Learning Rate: {config["train"]["learning_rate"]}")
    print(f"  Weight Decay: {onfig["train"]["weight_decay"]}")
    print("\n[Loss Function Parameters]")
    print(f"  Loss Alpha: {config["train"]["loss_alpha"]}")
    print(f"  Loss Beta: {config["train"]["loss_beta"]}")
    print("\n[Other Settings]")
    print(f"  Input Dimension: {input_dim}")
    print(f"  Number of Joints: {num_joints}")
    print(f"  Number of Dimensions: {num_dims}")
    print("---" * 20)