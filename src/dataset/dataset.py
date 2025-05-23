# data_loader/dataset.py
import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd

class HasRiskDataset(Dataset):
    """
    猪场数据集类，用于LSTM模型训练
    
    Args:
        features (numpy.ndarray): 特征序列数据，形状为 (样本数, 序列长度, 特征数)
        labels (numpy.ndarray): 标签数据，形状为 (样本数,)
        transform (callable, optional): 可选的数据转换函数
    """
    def __init__(self, df: pd.DataFrame, label: list):
        self.df = df
        self.label = label

        
    def __len__(self):
        """返回数据集大小"""
        return len(self.df)
    
    def __getitem__(self, idx):
        """获取单个样本"""
        row = self.df.iloc[idx]
        feature = row.drop(self.label).values
        label = row[self.label]
        
        # 转换为特定数据类型的张量
        feature_tensor = torch.tensor(feature, dtype=torch.float32)
        label_tensor = torch.tensor(label, dtype=torch.long)  # 转换为长整型
            
        return feature_tensor, label_tensor