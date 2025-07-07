# 深層学習モデルを構築するファイル
#
## スケーリング処理、ガウスノイズ等の処理は前処理で記述する
#
#
import pandas as pd 
import numpy as np
import torch 
import torch.nn as nn
import torch.nn.functional as F

class Transformer_Encoder(nn.Module):
    def __init__(self, input_dim, d_model, nhead, num_encoder_layers, num_joints, num_dims=3, dropout=0.1):
        super().__init__()

        # クラス属性としてnum_jointsを保存
        self.num_joints = num_joints
        self.num_dims = num_dims
        
        # 入力の特徴抽出を強化
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, d_model * 2),
            nn.LayerNorm(d_model * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model)
        )
        
        # Transformerネットワーク
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,  
            dropout=dropout,
            batch_first=True,
            norm_first=True
        )
        
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_encoder_layers
        )
        
        # 出力層の強化
        self.output_decoder = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.LayerNorm(d_model * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
            nn.ReLU(),
            nn.Linear(d_model, num_joints * num_dims)
        )
        
        # スケール係数（学習可能パラメータ）
        self.output_scale = nn.Parameter(torch.ones(1))
    
    def forward(self, x):

        # 特徴抽出
        features = self.feature_extractor(x)
        features = features.unsqueeze(1)
        
        # Transformer処理
        transformer_output = self.transformer_encoder(features)
        transformer_output = transformer_output.squeeze(1)
        
        # 出力生成とスケーリング
        output = self.output_decoder(transformer_output)
        output = output * self.output_scale  # 出力のスケーリング
        
        return output

class Skeleton_Loss(nn.Module):
    def __init__(self, alpha=1.0, beta=0.1):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        
    def forward(self, pred, target):
        # MSE損失
        mse_loss = F.mse_loss(pred, target)
        
        # 変化量の損失
        motion_loss = F.mse_loss(
            pred[1:] - pred[:-1],
            target[1:] - target[:-1]
        )
        
        # 加速度の損失
        accel_loss = F.mse_loss(
            pred[2:] + pred[:-2] - 2 * pred[1:-1],
            target[2:] + target[:-2] - 2 * target[1:-1]
        )
        
        return self.alpha * mse_loss + self.beta * (motion_loss + accel_loss)
    

def train_Transformer_Encoder(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs, save_path, device):
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        
        for pressure, skeleton in train_loader:
            # データをGPUに移動
            pressure = pressure.to(device)
            skeleton = skeleton.to(device)
            
            optimizer.zero_grad()
            
            outputs = model(pressure)
            loss = criterion(outputs, skeleton)
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for pressure, skeleton in val_loader:
                # データをGPUに移動
                pressure = pressure.to(device)
                skeleton = skeleton.to(device)
                
                outputs = model(pressure)
                loss = criterion(outputs, skeleton)
                val_loss += loss.item()
        
        # 平均損失の計算
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        
        # スケジューラのステップ
        scheduler.step(avg_val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        
        print(f'Epoch {epoch+1}')
        print(f'Training Loss: {avg_train_loss:.4f}')
        print(f'Validation Loss: {avg_val_loss:.4f}')
        print(f'Learning Rate: {current_lr:.6f}')
        
        # モデルの保存
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_val_loss': best_val_loss,
            }
            torch.save(checkpoint, save_path)
            print(f'Model saved at epoch {epoch+1}')
        
        print('-' * 60)


def load_Transformer_Encoder(model, optimizer, scheduler, checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    epoch = checkpoint['epoch']
    best_val_loss = checkpoint['best_val_loss']

    return model, optimizer, scheduler, epoch, best_val_loss


def save_predictions(predictions, model):
    # 予測結果をデータフレームに変換
    num_joints = predictions.shape[1] // 3
    columns = []
    for i in range(num_joints):
        columns.extend([f'X.{i}', f'Y.{i}', f'Z.{i}'])

    output_file='./output/predicted_skeleton.csv'       # 使用したモデルと実行日時のデータをファイル名に含める
    
    df_predictions = pd.DataFrame(predictions, columns=columns)
    df_predictions.to_csv(output_file, index=False)
    print(f"Predictions saved to {output_file}")


# モデルの推論
def predict_Transformer_Encoder(model, pressure_data):
    model.eval()
    with torch.no_grad():
        pressure_tensor = torch.FloatTensor(pressure_data)
        predictions = model(pressure_tensor)
    return predictions.numpy()