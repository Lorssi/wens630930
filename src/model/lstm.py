import torch
import torch.nn as nn
import torch.nn.functional as F

from config import EMBEDDING_SIZE

def get_activate_func(activate_func='relu', param: dict={}):
    if activate_func == 'relu':
        return nn.ReLU()
    elif activate_func == 'leaky_relu':
        negative_slope = param.get('negative_slope', 1e-2)
        return nn.LeakyReLU(negative_slope=negative_slope)
    else:
        raise ValueError("Unknown activate_func")

def make_mlp_layers(mlp_input_dim, hidden_dims, mlp_output_dim, activate_func='relu', **param):
    mlp_layers = nn.Sequential()
    dropout_rate = param.get('dropout_rate', None)
    if len(hidden_dims) == 0:
        mlp_layers.add_module("output", nn.Linear(mlp_input_dim, mlp_output_dim))
    else:
        mlp_layers.add_module("input", nn.Linear(mlp_input_dim, hidden_dims[0]))
        mlp_layers.add_module("activate0", get_activate_func(activate_func, param))
        if dropout_rate is not None:
            mlp_layers.add_module("dropout0", nn.Dropout(p=dropout_rate))
        # zip([512, 128, 64, 32], [128, 64, 32, 8]) -> (512,128), (128, 64), (64, 32), (32, 8)
        for i, (input_dim, output_dim) in enumerate(zip(hidden_dims[:-1], hidden_dims[1:])):
            mlp_layers.add_module("linear{}".format(i+1), nn.Linear(input_dim, output_dim))
            mlp_layers.add_module("activate{}".format(i+1), get_activate_func(activate_func, param))
            if dropout_rate is not None:
                mlp_layers.add_module("dropout{}".format(i+1), nn.Dropout(p=dropout_rate))

        mlp_layers.add_module("output", nn.Linear(hidden_dims[-1], mlp_output_dim))
        
    return mlp_layers

class FeatureInteraction(nn.Module):

    def __init__(self, emb_size=EMBEDDING_SIZE):
        super().__init__()

        self.emb_size = emb_size
        self.bn = nn.BatchNorm1d(self.emb_size)

    def forward(self, emb_stk):
        # emb_stk = [field_num, batch_size, emb_size]
        emb_stk = emb_stk.permute((1, 0, 2))
        # emb_stk = [batch_size, field_num, emb_size]
        emb_cross = 1 / 2 * (
                torch.pow(torch.sum(emb_stk, dim=1), 2) - torch.sum(torch.pow(emb_stk, 2), dim=1)
        ) # dim = 1, 沿field_num也就是特征维度求和
        # emb_cross = [batch_size, emb_size]
        return self.bn(emb_cross)
    
class Has_Risk_NFM_LSTM_MultiLabel_7d1Linear(nn.Module):
    """
    NFM模型用于猪场流产风险分类预测，支持特征交互，
    使用LSTM处理过去7天的流产率时序数据
    
    Args:
        params (dict): 配置参数字典，包含以下键:
            - model_discrete_columns: 离散特征列列表
            - model_continuous_columns: 连续特征列列表
            - dropout: Dropout概率
            - class_num: 每个离散特征的类别数量字典
    """
    def __init__(self, params: dict): 
        super(Has_Risk_NFM_LSTM_MultiLabel_7d1Linear, self).__init__()
        
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
        self.embedding_dim = EMBEDDING_SIZE
        
        # 为离散特征创建嵌入层
        self.embeddings = nn.ModuleDict()
        for feat in self.discrete_cols:
            if feat in params.keys():
                self.embeddings[feat] = nn.Embedding(
                    num_embeddings=int(params[feat] + 1),
                    embedding_dim=int(self.embedding_dim)
                )
        
        # 为需处理的连续特征创建变换
        self.continuous_transforms = nn.ModuleDict()
        
        # 定义过去7天的流产率特征
        self.past_7d_abortion_features = [f'abortion_rate_diff2_{day}d' for day in range(5, 0, -1)] 
        
        # 为流产率特征创建一个单独的线性变换
        self.abortion_rate_transform = nn.Linear(1, self.embedding_dim)
        
        # 创建LSTM处理流产率时序数据
        self.lstm = nn.LSTM(
            input_size=self.embedding_dim,
            hidden_size=self.embedding_dim,
            num_layers=1,
            batch_first=True
        )
        
        # 为其他连续特征创建变换
        for feat in self.continuous_cols:
            if feat not in self.past_7d_abortion_features:
                self.continuous_transforms[feat] = nn.Linear(1, self.embedding_dim)
        
        # 二阶特征交互层
        self.feature_interaction = FeatureInteraction(emb_size=self.embedding_dim)
        
        # 输出层
        self.mlp = make_mlp_layers(mlp_input_dim=EMBEDDING_SIZE, # 128
                                   hidden_dims=[128,64,32,8],
                                   mlp_output_dim=3,
                                   )
        
    
    def forward(self, x):
        """
        前向传播函数
        
        Args:
            x (torch.Tensor): 输入特征张量，特征和掩码交替排列
            
        Returns:
            torch.Tensor: 模型输出，形状为 [batch_size, 3]
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
                    
                    feat_emb = self.embeddings[feat](feat_data)
                    # 应用掩码
                    feat_emb = feat_emb * mask_data
                    embeddings_list.append(feat_emb)
        
        # 处理普通连续特征（非流产率特征）
        for feat in self.continuous_cols:
            if feat not in self.past_7d_abortion_features and feat in self.continuous_transforms:
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
        
        # 特殊处理过去7天流产率特征 - 使用LSTM
        abortion_seq = []
        abortion_mask_seq = []
        
        # 收集7天流产率数据和掩码
        for feat in self.past_7d_abortion_features:
            if feat in self.feature_indices:
                feat_idx = self.feature_indices[feat]
                mask_idx = feat_idx + 1
                
                if feat_idx < x.size(1) and mask_idx < x.size(1):
                    # 获取特征值和掩码
                    feat_data = x[:, feat_idx].float().view(batch_size, 1)
                    mask_data = x[:, mask_idx].float().view(batch_size, 1)
                    
                    abortion_seq.append(feat_data)
                    abortion_mask_seq.append(mask_data)
        
        if abortion_seq:
            # 将7天数据堆叠成序列 [batch_size, 7, 1]
            abortion_seq_tensor = torch.stack(abortion_seq, dim=1)
            abortion_mask_tensor = torch.stack(abortion_mask_seq, dim=1)
            
            # 对每个时间步应用相同的线性变换 [batch_size, 7, embedding_dim]
            abortion_seq_emb = self.abortion_rate_transform(abortion_seq_tensor)
            
            # 应用掩码
            abortion_seq_emb = abortion_seq_emb * abortion_mask_tensor.expand_as(abortion_seq_emb)
            
            # 通过LSTM处理序列
            lstm_out, _ = self.lstm(abortion_seq_emb)
            
            # 取最后一个时间步的输出作为流产率特征的表示 [batch_size, embedding_dim]
            abortion_feature = lstm_out[:, -1, :]
            
            # 将LSTM输出添加到嵌入列表
            embeddings_list.append(abortion_feature)
        
        # 检查是否有特征被成功处理
        if not embeddings_list:
            raise ValueError("没有特征被处理，请检查输入数据和特征配置")
        
        # 堆叠所有特征的嵌入向量 [field_num, batch_size, emb_size]
        stacked_embeddings = torch.stack(embeddings_list)
        
        # 二阶特征交互
        interaction_output = self.feature_interaction(stacked_embeddings)
        
        # 输入到MLP
        output = self.mlp(interaction_output)
        
        return output