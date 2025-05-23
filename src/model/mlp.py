import torch
import torch.nn as nn
import torch.nn.functional as F

from config import EMBEDDING_SIZE

class Has_Risk_MLP(nn.Module):
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
    def __init__(self, params: dict): 
        super(Has_Risk_MLP, self).__init__()
        
        # 配置特征列
        self.discrete_cols = params['model_discrete_columns'] if params['model_discrete_columns'] is not None else ['pigfarm_dk', 'is_single', 'month']
        self.continuous_cols = params['model_continuous_columns']  if params['model_continuous_columns']  is not None else ['intro_num']
        self.dropout = params['dropout'] if params['dropout'] is not None else 0.2
        
        # 特征位置映射
        self.feature_indices = {}
        idx = 0
        for col in self.discrete_cols + self.continuous_cols:
            self.feature_indices[col] = idx
            idx += 2
        
        # 嵌入维度
        self.embedding_dim = 8
        
        # 为离散特征创建嵌入层
        self.embeddings = nn.ModuleDict()
        for feat in self.discrete_cols:
            if feat in params.keys():
                self.embeddings[feat] = nn.Embedding(
                    num_embeddings=int(params[feat] + 2),
                    embedding_dim=int(self.embedding_dim)
                )
        
        # 为需处理的连续特征创建变换
        self.continuous_transforms = nn.ModuleDict()
        for feat in self.continuous_cols:
            self.continuous_transforms[feat] = nn.Linear(1, self.embedding_dim)
        
        # 特征维度计算
        embedded_dim = self.embedding_dim * (len(self.discrete_cols) + len(self.continuous_cols))
        total_feature_dim = embedded_dim
               
        # 输出层
        input_size = total_feature_dim
        self.mlp = nn.Sequential(
            nn.Linear(input_size, 16),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(16, 4)
        )
        
    
    def forward(self, x):
        """
        前向传播函数
        
        Args:
            x (torch.Tensor): 输入特征张量，特征和掩码交替排列
            
        Returns:
            torch.Tensor: 模型输出，形状为 [batch_size, 2]
        """
        batch_size = x.size(0)
        embeddings_list = []
        
        # 处理离散特征
        for feat in self.discrete_cols:
            if feat in self.embeddings:
                # 使用预先定义的特征索引
                feat_idx = self.feature_indices[feat]
                mask_idx = feat_idx + 1  # 掩码总是紧跟在特征后面
                
                # 确保索引在有效范围内
                if feat_idx < x.size(1) and mask_idx < x.size(1):
                    # 获取特征值和掩码
                    feat_data = x[:, feat_idx].long()
                    mask_data = x[:, mask_idx].float().view(batch_size, 1)
                    
                    # 通过嵌入层处理
                    feat_emb = self.embeddings[feat](feat_data)
                    # 应用掩码
                    feat_emb = feat_emb * mask_data
                    embeddings_list.append(feat_emb)
        
        # 处理连续特征
        for feat in self.continuous_cols:
            if feat in self.continuous_transforms:
                # 使用预先定义的特征索引
                feat_idx = self.feature_indices[feat]
                mask_idx = feat_idx + 1  # 掩码总是紧跟在特征后面
                
                # 确保索引在有效范围内
                if feat_idx < x.size(1) and mask_idx < x.size(1):
                    # 获取特征值和掩码
                    feat_data = x[:, feat_idx].float().view(batch_size, 1)
                    mask_data = x[:, mask_idx].float().view(batch_size, 1)
                    
                    # 通过线性层处理
                    feat_emb = self.continuous_transforms[feat](feat_data)
                    # 应用掩码
                    feat_emb = feat_emb * mask_data.expand_as(feat_emb)
                    embeddings_list.append(feat_emb)
        
        # 检查是否有特征被成功处理
        if not embeddings_list:
            raise ValueError("没有特征被处理，请检查输入数据和特征配置")
        
        # 拼接所有特征的嵌入向量
        combined_features = torch.cat(embeddings_list, dim=1)
        
        # 输入到MLP
        output = self.mlp(combined_features)
        
        return output
    