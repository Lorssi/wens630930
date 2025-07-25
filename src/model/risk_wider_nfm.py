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
    
class WideLayer(nn.Module):
    def __init__(self, params: dict):
        super(WideLayer, self).__init__()
        
        self.discrete_cols = params['model_discrete_columns']
        self.continuous_cols = params['model_continuous_columns']  
        self.dropout = params['dropout'] if params['dropout'] is not None else 0.2
        
        # 特征位置映射
        self.feature_indices = {}
        idx = 0
        for col in self.discrete_cols + self.continuous_cols:
            self.feature_indices[col] = idx
            idx += 2

        # 为离散特征创建嵌入层
        self.embeddings = nn.ModuleDict()
        for feat in self.discrete_cols:
            if feat in params.keys():
                self.embeddings[feat] = nn.Embedding(
                    num_embeddings=int(params[feat] + 1),
                    embedding_dim=3
                )

        # 连续特征转换层 - 将所有连续特征合并后转换为3维向量
        self.continuous_transform = nn.Linear(len(self.continuous_cols), 3)

    def forward(self, x):
        batch_size = x.size(0)
        wide_features = []
        
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
                    
                    feat_emb = self.embeddings[feat](feat_data)  # [batch_size, 3]
                    # 应用掩码
                    feat_emb = feat_emb * mask_data
                    wide_features.append(feat_emb)

        # 处理连续特征
        continuous_features = []
        for feat in self.continuous_cols:
            # 使用预先定义的特征索引
            feat_idx = self.feature_indices[feat]
            mask_idx = feat_idx + 1  # 掩码总是紧跟在特征后面
            
            # 确保索引在有效范围内
            if feat_idx < x.size(1) and mask_idx < x.size(1):
                # 提取特征值和掩码值
                feat_value = x[:, feat_idx].unsqueeze(1)  # [batch_size, 1]
                mask_value = x[:, mask_idx].unsqueeze(1)  # [batch_size, 1]
                
                # 应用掩码，对于掩码为0的位置，将特征值置为0
                masked_feat = feat_value * mask_value
                
                # 将处理后的特征添加到列表
                continuous_features.append(masked_feat)

        # 如果有连续特征，则处理
        if continuous_features and len(continuous_features) > 0:
            continuous_concat = torch.cat(continuous_features, dim=1)  # [batch_size, num_features]
            continuous_output = self.continuous_transform(continuous_concat)  # [batch_size, 3]
            wide_features.append(continuous_output)

        # 如果没有有效特征，返回零向量
        if not wide_features:
            return torch.zeros(batch_size, 3, device=x.device)

        # 将所有特征嵌入求和
        if len(wide_features) == 1:
            return wide_features[0]
        else:
            # 所有特征维度都是[batch_size, 3]
            wide_concat = torch.stack(wide_features, dim=1)  # [batch_size, num_features, 3]
            return torch.sum(wide_concat, dim=1)  # [batch_size, 3]

class Has_Risk_NFM_MultiLabel_Wider(nn.Module):
    """
    NFM模型用于猪场流产风险分类预测，支持特征交互
    
    Args:
        params (dict): 配置参数字典，包含以下键:
            - model_discrete_columns: 离散特征列列表
            - model_continuous_columns: 连续特征列列表
            - dropout: Dropout概率
            - class_num: 每个离散特征的类别数量字典
    """
    def __init__(self, params: dict): 
        super(Has_Risk_NFM_MultiLabel_Wider, self).__init__()
        
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
        past_7d_abortion_features = [f'abortion_rate_past_{day + 1}d' for day in range(7)]
        for feat in self.continuous_cols:
            if feat in past_7d_abortion_features:
                self.continuous_transforms['abortion_all_7d'] = nn.Linear(1, self.embedding_dim)
            else:
                self.continuous_transforms[feat] = nn.Linear(1, self.embedding_dim)
        
        # 二阶特征交互层
        self.feature_interaction = FeatureInteraction(emb_size=self.embedding_dim)
        # wider 
        self.wide_layer = WideLayer(params=params)
        
        # 输出层
        self.mlp = make_mlp_layers(mlp_input_dim=EMBEDDING_SIZE, # 128
                                   hidden_dims=[32, 16],
                                #    hidden_dims=[512, 128, 32, 8],
                                   mlp_output_dim=EMBEDDING_SIZE,
                                   dropout_rate=self.dropout)
        
        self.linear_layer = nn.Linear(2 * EMBEDDING_SIZE, 3) # 3是wide的输出维度，8是mlp的输出维度

    def forward(self, x):
        """
        前向传播函数
        
        Args:
            x (torch.Tensor): 输入特征张量，特征和掩码交替排列
            
        Returns:
            torch.Tensor: 模型输出，形状为 [batch_size, 4]
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
        
        # 处理连续特征
        past_7d_abortion_features = [f'abortion_rate_past_{day + 1}d' for day in range(7)]
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
            elif feat in past_7d_abortion_features:
                # 使用预先定义的特征索引
                feat_idx = self.feature_indices[feat]
                mask_idx = feat_idx + 1  # 掩码总是紧跟在特征后面
                
                # 确保索引在有效范围内
                if feat_idx < x.size(1) and mask_idx < x.size(1):
                    # 获取特征值和掩码
                    feat_data = x[:, feat_idx].float().view(batch_size, 1)
                    mask_data = x[:, mask_idx].float().view(batch_size, 1)
                    
                    # 通过线性层处理
                    feat_emb = self.continuous_transforms['abortion_all_7d'](feat_data)
                    # 应用掩码
                    feat_emb = feat_emb * mask_data.expand_as(feat_emb)
                    embeddings_list.append(feat_emb)
        
        # 检查是否有特征被成功处理
        if not embeddings_list:
            raise ValueError("没有特征被处理，请检查输入数据和特征配置")
        
        # 堆叠所有特征的嵌入向量 [field_num, batch_size, emb_size]
        stacked_embeddings = torch.stack(embeddings_list)
        
        # 二阶特征交互
        interaction_output = self.feature_interaction(stacked_embeddings)  
        # 输入到MLP
        mlp_output = self.mlp(interaction_output)
        output = self.linear_layer(torch.concat([interaction_output, mlp_output], dim=1)) # [batch_size, 3]

        wide = self.wide_layer(x) # [batch_size, 3]
        # 将wide和mlp输出相加
        output = output + wide
        
        return output