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

class TaskHead_Days(nn.Module):
    def __init__(self, input_dim=64, output_dim=8):
        super().__init__()
        self.head = nn.Linear(input_dim, output_dim)  # 任务特定层
    
    def forward(self, x):

        return self.head(x)

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

class Days_NFM(nn.Module):
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
        super(Days_NFM, self).__init__()
        
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
                    num_embeddings=int(params[feat] + 2),
                    embedding_dim=int(self.embedding_dim)
                )
        
        # 为需处理的连续特征创建变换
        self.continuous_transforms = nn.ModuleDict()
        for feat in self.continuous_cols:
            self.continuous_transforms[feat] = nn.Linear(1, self.embedding_dim)
        
        # 二阶特征交互层
        self.feature_interaction = FeatureInteraction(emb_size=self.embedding_dim)
        
        # 输出层
        self.mlp = make_mlp_layers(mlp_input_dim=EMBEDDING_SIZE, # 128
                                   hidden_dims=[128, 64, 32],
                                #    hidden_dims=[512, 128, 32, 8],
                                   mlp_output_dim=16)

        # self.task_head_has_risk = TaskHead_HasRisk(input_dim=16, output_dim=3)
        self.task_head_days_1_7 = TaskHead_Days(input_dim=16, output_dim=8)
        self.task_head_days_8_14 = TaskHead_Days(input_dim=16, output_dim=8)
        self.task_head_days_15_21 = TaskHead_Days(input_dim=16, output_dim=8)
        
    
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
        
        # 堆叠所有特征的嵌入向量 [field_num, batch_size, emb_size]
        stacked_embeddings = torch.stack(embeddings_list)
        
        # 二阶特征交互
        interaction_output = self.feature_interaction(stacked_embeddings)
        
        # 输入到MLP
        output = self.mlp(interaction_output)

        output = F.relu(output)

        days_1_7_output = self.task_head_days_1_7(output)
        days_8_14_output = self.task_head_days_8_14(output)
        days_15_21_output = self.task_head_days_15_21(output)
        
        return days_1_7_output, days_8_14_output, days_15_21_output