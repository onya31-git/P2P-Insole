# 深層学習モデルを構築するファイル
#
## スケーリング処理、ガウスノイズ等の処理は前処理で記述する
## train_test_splitの処理をtrainファイルに移動する
## データローダー, データセットの処理はutilsファイルに移動する

import pandas as pd 
import numpy as np
import torch 
import torch.nn as nn
import torch.nn.functional as F

