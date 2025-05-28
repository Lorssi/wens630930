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
    def __init__(self, df: pd.DataFrame, label: str):
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
    
class DaysDataset(Dataset):
    """
    多任务数据集类，用于多任务学习模型训练
    
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
        labels = row[self.label].values
            
        return feature, labels
    
class MultiTaskDataset(Dataset):
    """
    多任务数据集类，用于多任务学习模型训练
    
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
        has_risk_label = row[self.label[0]]
        days_1_7 = row[self.label[1]]
        days_8_14 = row[self.label[2]]
        days_15_21 = row[self.label[3]]
        
        # 转换为特定数据类型的张量
        feature_tensor = torch.tensor(feature, dtype=torch.float32)
        has_risk_label_tensor = torch.tensor(has_risk_label, dtype=torch.long)  # 转换为长整型
        days_1_7_tensor = torch.tensor(days_1_7, dtype=torch.long)
        days_8_14_tensor = torch.tensor(days_8_14, dtype=torch.long)
        days_15_21_tensor = torch.tensor(days_15_21, dtype=torch.long)
            
        return feature_tensor, has_risk_label_tensor, days_1_7_tensor, days_8_14_tensor, days_15_21_tensor
    
class MultiTaskAndMultiLabelDataset(Dataset):
    """
    多任务数据集类，用于多任务学习模型训练
        
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
        label = row[self.label].values
                          
        return feature, label