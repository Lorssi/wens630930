import torch
import torch.nn as nn
import torch.nn.functional as F

from config import EMBEDDING_SIZE

class HasRiskLSTM(nn.Module):
    """
    LSTM模型用于猪场流产风险二分类预测，支持动态扩展特征
    
    Args:
        input_size (int): 原始输入特征维度（主要为兼容性保留）
        hidden_size (int): LSTM隐藏层大小
        num_layers (int): LSTM层数
        dropout (float): Dropout概率
        bidirectional (bool): 是否使用双向LSTM
        class_num (dict): 每个离散特征的类别数量字典
    """
    def __init__(self, hidden_size=32, num_layers=1, dropout=0.3,
                 bidirectional=True, class_num=None, model_discrete_columns=None,
                 model_continuous_columns=None, model_direct_columns=None):
        super(HasRiskLSTM, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        
        # 配置特征列
        self.discrete_cols = model_discrete_columns if model_discrete_columns is not None else ['pigfarm_dk', 'is_single', 'month']
        self.continuous_cols = model_continuous_columns if model_continuous_columns is not None else ['intro_num']
        self.direct_cols = model_direct_columns if model_direct_columns is not None else ['month_sin', 'month_cos']
        
        # 特征位置映射
        self.feature_indices = {}
        idx = 0
        for col in self.discrete_cols + self.continuous_cols + self.direct_cols:
            self.feature_indices[col] = idx
            idx += 2
        
        # 嵌入维度
        self.embedding_dim = 8
        
        # 为离散特征创建嵌入层
        self.embeddings = nn.ModuleDict()
        if class_num is not None:
            for feat in self.discrete_cols:
                if feat in class_num:
                    self.embeddings[feat] = nn.Embedding(
                        num_embeddings=int(class_num[feat] + 2),
                        embedding_dim=int(self.embedding_dim)
                    )
        
        # 为需处理的连续特征创建变换
        self.continuous_transforms = nn.ModuleDict()
        for feat in self.continuous_cols:
            self.continuous_transforms[feat] = nn.Linear(1, self.embedding_dim)
        
        # 特征维度计算
        embedded_dim = self.embedding_dim * (len(self.discrete_cols) + len(self.continuous_cols))
        direct_dim = len(self.direct_cols)  # 每个直接特征是1维
        total_feature_dim = embedded_dim + direct_dim
        
        # 特征降维
        reduced_dim = 32
        self.feature_reduction = nn.Sequential(
            nn.Linear(total_feature_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, reduced_dim)
        )
        
        # LSTM层
        self.lstm = nn.LSTM(
            input_size=reduced_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.0,  # 单层LSTM不使用dropout
            bidirectional=bidirectional
        )
        
        # 输出层
        fc_input_size = hidden_size * self.num_directions
        self.fc = nn.Sequential(
            nn.Linear(fc_input_size, 16),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(16, 2)
        )
        
    
    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        
        # 处理特征
        processed_features = []  # 嵌入特征
        direct_features = []     # 直接特征
        
        # 处理离散特征
        for feat in self.discrete_cols:
            feat_idx = self.feature_indices[feat]
            mask_idx = feat_idx + 1
            
            feat_values = x[:, :, feat_idx].long() 
            feat_mask = x[:, :, mask_idx]
            
            emb = self.embeddings[feat](feat_values)
            masked_emb = emb * feat_mask.unsqueeze(-1)
            processed_features.append(masked_emb)
        
        # 处理需变换的连续特征
        for feat in self.continuous_cols:
            feat_idx = self.feature_indices[feat]
            mask_idx = feat_idx + 1
            
            feat_values = x[:, :, feat_idx].unsqueeze(-1)
            feat_mask = x[:, :, mask_idx].unsqueeze(-1)
            
            transformed = self.continuous_transforms[feat](feat_values)
            masked_transformed = transformed * feat_mask
            processed_features.append(masked_transformed)
        
        # 处理直接输入的特征
        for feat in self.direct_cols:
            feat_idx = self.feature_indices[feat]
            mask_idx = feat_idx + 1
            
            feat_values = x[:, :, feat_idx].unsqueeze(-1)
            feat_mask = x[:, :, mask_idx].unsqueeze(-1)
            
            masked_values = feat_values * feat_mask
            direct_features.append(masked_values)
        
        # 合并特征
        if processed_features:
            embedded_features = torch.cat(processed_features, dim=-1)
        else:
            embedded_features = torch.zeros((batch_size, seq_len, 0), device=x.device)
            
        if direct_features:
            combined_direct = torch.cat(direct_features, dim=-1)
        else:
            combined_direct = torch.zeros((batch_size, seq_len, 0), device=x.device)
        
        # 拼接所有特征
        all_features = torch.cat([embedded_features, combined_direct], dim=-1) # [batch_size, seq_len, total_feature_dim]
        
        # 应用降维
        reshaped = all_features.reshape(-1, all_features.size(-1))
        reduced = self.feature_reduction(reshaped)
        reduced = reduced.reshape(batch_size, seq_len, -1)
        
        # LSTM处理
        out, (hn, cn) = self.lstm(reduced)
        
        # 取最后时间步
        lstm_out = out[:, -1, :]
        
        # 分类器
        out = self.fc(lstm_out)
        
        return out

# 相加取平均
# class HasRiskLSTM(nn.Module):
#     """
#     LSTM模型用于猪场流产风险二分类预测，支持动态扩展特征
    
#     Args:
#         input_size (int): 原始输入特征维度（主要为兼容性保留）
#         hidden_size (int): LSTM隐藏层大小
#         num_layers (int): LSTM层数
#         dropout (float): Dropout概率
#         bidirectional (bool): 是否使用双向LSTM
#         class_num (dict): 每个离散特征的类别数量字典
#     """
#     def __init__(self, hidden_size=64, num_layers=2, dropout=0.2,
#                  bidirectional=True, class_num=None, model_discrete_columns=None,
#                  model_continuous_columns=None):
#         super(HasRiskLSTM, self).__init__()
        
#         self.hidden_size = hidden_size
#         self.num_layers = num_layers
#         self.bidirectional = bidirectional
#         self.num_directions = 2 if bidirectional else 1
        
#         # 配置特征列
#         self.discrete_cols = model_discrete_columns if model_discrete_columns is not None else ['pigfarm_dk', 'is_single', 'month']
#         self.continuous_cols = model_continuous_columns if model_continuous_columns is not None else ['intro_num']
        
#         # 特征位置映射（特征名 -> 在输入tensor中的位置索引）
#         self.feature_indices = {}
#         idx = 0
#         for col in self.discrete_cols + self.continuous_cols:
#             self.feature_indices[col] = idx
#             idx += 2  # 每个特征后面跟着一个mask
        
#         # 嵌入维度
#         self.embedding_dim = EMBEDDING_SIZE
        
#         # 为每个离散特征创建嵌入层
#         self.embeddings = nn.ModuleDict()
#         if class_num is not None:
#             for feat in self.discrete_cols:
#                 if feat in class_num:
#                     self.embeddings[feat] = nn.Embedding(
#                         num_embeddings=int(class_num[feat] + 2),  # +1 用于处理未知类别
#                         embedding_dim=int(self.embedding_dim)
#                     )
        
#         # 为每个连续特征创建线性变换
#         self.continuous_transforms = nn.ModuleDict()
#         for feat in self.continuous_cols:
#             self.continuous_transforms[feat] = nn.Linear(1, self.embedding_dim)
        
#         # 特征处理后的总维度
#         total_features = len(self.discrete_cols) + len(self.continuous_cols)
#         feature_dim = self.embedding_dim * total_features
        
#         # LSTM层
#         self.lstm = nn.LSTM(
#             input_size=self.embedding_dim,
#             hidden_size=hidden_size,
#             num_layers=num_layers,
#             batch_first=True,
#             dropout=dropout if num_layers > 1 else 0,
#             bidirectional=bidirectional
#         )
        
#         # 全连接输出层
#         fc_input_size = hidden_size * self.num_directions

#         self.fc = nn.Sequential(
#             nn.Linear(fc_input_size, 8),
#             nn.ReLU(),
#             nn.Dropout(dropout),
#             nn.Linear(8, 2),
#         )
        
#         self._init_weights()

#     def _init_weights(self):
#         for name, param in self.lstm.named_parameters():
#             if 'weight_ih' in name:
#                 nn.init.xavier_uniform_(param.data)
#             elif 'weight_hh' in name:
#                 nn.init.orthogonal_(param.data)
#             elif 'bias' in name:
#                 nn.init.constant_(param.data, 0)
    
#     def forward(self, x):
#         """
#         前向传播
        
#         Args:
#             x (torch.Tensor): 输入序列，形状为 [batch_size, seq_len, feature_dim]
#                               其中feature_dim包含特征和对应的掩码
        
#         Returns:
#             torch.Tensor: 二分类预测概率
#         """
#         batch_size, seq_len, _ = x.shape
        
#         # 处理所有特征
#         processed_features = []
        
#         # 处理离散特征
#         for feat in self.discrete_cols:
#             feat_idx = self.feature_indices[feat]
#             mask_idx = feat_idx + 1
            
#             # 提取特征和掩码
#             feat_values = x[:, :, feat_idx].long()  # [batch_size, seq_len]
#             feat_mask = x[:, :, mask_idx]  # [batch_size, seq_len]
            
#             # 应用嵌入和掩码
#             emb = self.embeddings[feat](feat_values)  # [batch_size, seq_len, embedding_dim]
#             masked_emb = emb * feat_mask.unsqueeze(-1)  # 应用掩码
#             processed_features.append(masked_emb)
        
#         # 处理连续特征
#         for feat in self.continuous_cols:
#             feat_idx = self.feature_indices[feat] # intro_num 索引
#             mask_idx = feat_idx + 1 # intro_num_mask 索引
            
#             # 提取特征和掩码
#             feat_values = x[:, :, feat_idx].unsqueeze(-1)  # [batch_size, seq_len, 1] 
#             feat_mask = x[:, :, mask_idx].unsqueeze(-1)  # [batch_size, seq_len, 1]
            
#             # 应用线性变换和掩码
#             transformed = self.continuous_transforms[feat](feat_values)  # [batch_size, seq_len, embedding_dim]
#             masked_transformed = transformed * feat_mask 
#             processed_features.append(masked_transformed)
        
#         # 合并所有处理后的特征
#         # combined_features = torch.cat(processed_features, dim=-1)  # [batch_size, seq_len, feature_dim]
#         stacked_features = torch.stack(processed_features, dim=2) # [batch_size, seq_len, num_features, embedding_dim]
#         # 计算非零特征的掩码（避免除以零）
#         feature_mask = (stacked_features.sum(dim=-1) != 0).float()  # [batch_size, seq_len, num_features]
#         feature_count = feature_mask.sum(dim=2, keepdim=True).clamp(min=1.0)  # 至少为1，避免除以0
        
#         # 对特征求和后取平均
#         combined_features = stacked_features.sum(dim=2) / feature_count  # [batch_size, seq_len, embedding_dim]
        
#         # LSTM前向传播
#         out, (hn, cn) = self.lstm(combined_features)
        
#         # 只取最后一个时间步的输出
#         lstm_out = out[:, -1, :]
        
#         # 全连接层
#         out = self.fc(lstm_out)  # [batch_size, 2]
  
#         return out

# 原始
# class HasRiskLSTM(nn.Module):
#     """
#     LSTM模型用于猪场流产风险二分类预测，支持动态扩展特征
    
#     Args:
#         input_size (int): 原始输入特征维度（主要为兼容性保留）
#         hidden_size (int): LSTM隐藏层大小
#         num_layers (int): LSTM层数
#         dropout (float): Dropout概率
#         bidirectional (bool): 是否使用双向LSTM
#         class_num (dict): 每个离散特征的类别数量字典
#     """
#     def __init__(self, hidden_size=64, num_layers=2, dropout=0.2,
#                  bidirectional=True, class_num=None, model_discrete_columns=None,
#                  model_continuous_columns=None):
#         super(HasRiskLSTM, self).__init__()
        
#         self.hidden_size = hidden_size
#         self.num_layers = num_layers
#         self.bidirectional = bidirectional
#         self.num_directions = 2 if bidirectional else 1
        
#         # 配置特征列
#         self.discrete_cols = model_discrete_columns if model_discrete_columns is not None else ['pigfarm_dk', 'is_single', 'month']
#         self.continuous_cols = model_continuous_columns if model_continuous_columns is not None else ['intro_num']
        
#         # 特征位置映射（特征名 -> 在输入tensor中的位置索引）
#         self.feature_indices = {}
#         idx = 0
#         for col in self.discrete_cols + self.continuous_cols:
#             self.feature_indices[col] = idx
#             idx += 2  # 每个特征后面跟着一个mask
        
#         # 嵌入维度
#         self.embedding_dim = 8
        
#         # 为每个离散特征创建嵌入层
#         self.embeddings = nn.ModuleDict()
#         if class_num is not None:
#             for feat in self.discrete_cols:
#                 if feat in class_num:
#                     self.embeddings[feat] = nn.Embedding(
#                         num_embeddings=int(class_num[feat] + 2),  # +1 用于处理未知类别
#                         embedding_dim=int(self.embedding_dim)
#                     )
        
#         # 为每个连续特征创建线性变换
#         self.continuous_transforms = nn.ModuleDict()
#         for feat in self.continuous_cols:
#             self.continuous_transforms[feat] = nn.Linear(1, self.embedding_dim)
        
#         # 特征处理后的总维度
#         total_features = len(self.discrete_cols) + len(self.continuous_cols)
#         feature_dim = self.embedding_dim * total_features
        
#         # LSTM层
#         self.lstm = nn.LSTM(
#             input_size=feature_dim,
#             hidden_size=hidden_size,
#             num_layers=num_layers,
#             batch_first=True,
#             dropout=dropout if num_layers > 1 else 0,
#             bidirectional=bidirectional
#         )
        
#         # 全连接输出层
#         fc_input_size = hidden_size * self.num_directions

#         self.fc = nn.Sequential(
#             nn.Linear(fc_input_size, 16),
#             nn.BatchNorm1d(16),  # 添加BN层
#             nn.ReLU(),
#             nn.Dropout(dropout),
#             nn.Linear(16, 2)
#         )
    
#     def forward(self, x):
#         """
#         前向传播
        
#         Args:
#             x (torch.Tensor): 输入序列，形状为 [batch_size, seq_len, feature_dim]
#                               其中feature_dim包含特征和对应的掩码
        
#         Returns:
#             torch.Tensor: 二分类预测概率
#         """
#         batch_size, seq_len, _ = x.shape
        
#         # 处理所有特征
#         processed_features = []
        
#         # 处理离散特征
#         for feat in self.discrete_cols:
#             feat_idx = self.feature_indices[feat]
#             mask_idx = feat_idx + 1
            
#             # 提取特征和掩码
#             feat_values = x[:, :, feat_idx].long()  # [batch_size, seq_len]
#             feat_mask = x[:, :, mask_idx]  # [batch_size, seq_len]
            
#             # 应用嵌入和掩码
#             emb = self.embeddings[feat](feat_values)  # [batch_size, seq_len, embedding_dim]
#             masked_emb = emb * feat_mask.unsqueeze(-1)  # 应用掩码
#             processed_features.append(masked_emb)
        
#         # 处理连续特征
#         for feat in self.continuous_cols:
#             feat_idx = self.feature_indices[feat] # intro_num 索引
#             mask_idx = feat_idx + 1 # intro_num_mask 索引
            
#             # 提取特征和掩码
#             feat_values = x[:, :, feat_idx].unsqueeze(-1)  # [batch_size, seq_len, 1] 
#             feat_mask = x[:, :, mask_idx].unsqueeze(-1)  # [batch_size, seq_len, 1]
            
#             # 应用线性变换和掩码
#             transformed = self.continuous_transforms[feat](feat_values)  # [batch_size, seq_len, embedding_dim]
#             masked_transformed = transformed * feat_mask 
#             processed_features.append(masked_transformed)
        
#         # 合并所有处理后的特征
#         combined_features = torch.cat(processed_features, dim=-1)  # [batch_size, seq_len, feature_dim]
        
#         # LSTM前向传播
#         out, (hn, cn) = self.lstm(combined_features)
        
#         # 只取最后一个时间步的输出
#         lstm_out = out[:, -1, :]
        
#         # 全连接层
#         out = self.fc(lstm_out)  # [batch_size, 2]
  
#         return out