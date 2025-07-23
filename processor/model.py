# 深層学習モデルを構築するファイル
#
# サマリーを表示できるようにする(実行時間、best loss、 エポック、等)
# 予測トレーニング終了時間を表示する
#
import pandas as pd 
import math
import time
import datetime
from tqdm import tqdm
import torch 
import torch.nn as nn
import torch.nn.functional as F

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super(PositionalEncoding, self).__init__()
        
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x の形状: [バッチサイズ, シーケンス長, d_model]
        x = x + self.pe[:, :x.size(1)]
        return x

class Transformer_Encoder(nn.Module):
    def __init__(self, input_dim, d_model, nhead, num_encoder_layers, num_joints, num_dims=3, dropout=0.1):
        super().__init__()

        self.num_joints = num_joints
        self.num_dims = num_dims

        # first layer
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, d_model),  # 次元を調整
            nn.LayerNorm(d_model),          # 学習を安定化
            nn.ReLU(),                      # 活性化関数 (Mishもあり)
            nn.Dropout(dropout),            # 過学習を防止
            nn.Linear(d_model, d_model))    # 次元を調整

        # positional encording
        self.positional_encoder = PositionalEncoding(d_model)

        # transformer encoder
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=d_model,              # モデルの次元(高いほどモデルの表現力が上がる)
                nhead=nhead,                  # Attention headの数
                dim_feedforward=d_model * 4,  # FeedForward layerの数(ここも入力から調整する必要があるかも)
                dropout=dropout,              # ドロップアウト正則化
                batch_first=True,             # 
                norm_first=True),             # 

            num_layers=num_encoder_layers
        )
        
        # output layer
        self.output_decoder = nn.Sequential(
            nn.Linear(d_model, d_model),                # 次元を調整
            nn.LayerNorm(d_model),                      # 学習の安定化
            nn.ReLU(),                                  # 活性化関数(Mishはいかが?)
            nn.Dropout(dropout),                        # ドロップアウト正則化
            nn.Linear(d_model, d_model),                # 次元を調整
            nn.ReLU(),                                  # 活性化関数
            nn.Linear(d_model, num_joints * num_dims))  # 最終出力層(関節点の出力)
        
        # scaling factor
        self.output_scale = nn.Parameter(torch.ones(1)) 
    
    def forward(self, x):
        features = self.feature_extractor(x)                     # 特徴抽出
        features = self.positional_encoder(features)             # positional encording
        transformer_output = self.transformer_encoder(features)  # Transformer_encoder処理
        last_time_step_output = transformer_output[:, -1, :]     # シーケンスの最後の時点の情報を抽出
        output = self.output_decoder(last_time_step_output)      # 出力生成とスケーリング
        output = output * self.output_scale                      # 出力のスケーリング
        return output

class Skeleton_Loss(nn.Module):
    def __init__(self, alpha=1.0, beta=0.1):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        
    def forward(self, pred, target):
        mse_loss = F.mse_loss(pred, target)  # シンプルな MSE Loss
        return mse_loss
        # # 変化量の損失
        # motion_loss = F.mse_loss(
        #     pred[1:] - pred[:-1],
        #     target[1:] - target[:-1])
        
        # # 加速度の損失
        # accel_loss = F.mse_loss(
        #     pred[2:] + pred[:-2] - 2 * pred[1:-1],
        #     target[2:] + target[:-2] - 2 * target[1:-1])
        
        # return self.alpha * mse_loss + self.beta * (motion_loss + accel_loss)
    

def train_Transformer_Encoder(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs, save_path, device):
    best_val_loss = float('inf')

    # 開始時間の記録
    start_time = time.time()
    pre_time   = start_time
    now_time   = datetime.datetime.now()
    print(f"\n[train started at {now_time.strftime("%H:%M")}]")
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1} Train", leave=False)
        for pressure, skeleton in train_pbar:
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
        
        val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1} Val", leave=False)
        with torch.no_grad():
            for pressure, skeleton in val_pbar:
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

        # 経過時間の計算
        end_time = time.time()
        epoch_time_total_s   = end_time - pre_time
        elaps_time_total_s   = end_time - start_time
        epoch_time_m         = int(epoch_time_total_s//60)
        epoch_time_s         = int(epoch_time_total_s%60)
        elaps_time_m         = int(elaps_time_total_s//60)
        elaps_time_s         = int(elaps_time_total_s%60)

        # 予測完了時間の計算
        # est_comp_time = 

        print(f'------------ Epoch {epoch+1}/{num_epochs} ------------\n'
              f'Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}\n'
              f'LR        : {current_lr:.5f}\n'
              f'Time/epoch: {epoch_time_m}m {epoch_time_s}s | Total: {elaps_time_m}m {elaps_time_s}s') #\n'
              # f'Estimated completion time - {}:{}')
        
        pre_time = end_time
        
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
            print(f'>> Model saved at epoch {epoch+1}')


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