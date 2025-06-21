# main_train.py
# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.utils.data import DataLoader
import numpy as np
import os
import pandas as pd
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.utils.data import WeightedRandomSampler
from sklearn.metrics import roc_auc_score, precision_score, recall_score, accuracy_score
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.model_selection import train_test_split

# 从项目中导入模块
from configs.logger_config import logger_config
from configs.feature_config import ColumnsConfig, DataPathConfig
import config
from dataset.dataset import HasRiskDataset, MultiTaskAndMultiLabelDataset
from dataset.risk_prediction_index_sample_dataset import RiskPredictionIndexSampleDataset
from dataset.risk_prediction_feature_dataset import RiskPredictionFeatureDataset
from utils.logger import setup_logger
from utils.early_stopping import EarlyStopping
from utils.save_csv import save_to_csv, read_csv
from feature.gen_feature import FeatureGenerator
from feature.gen_label import LabelGenerator
from transform.transform import FeatureTransformer
from model.mlp import Has_Risk_MLP
from model.nfm import Has_Risk_NFM, Has_Risk_NFM_MultiLabel, Has_Risk_NFM_MultiLabel_7d1Linear
from model.risk_wider_nfm import Has_Risk_NFM_MultiLabel_Wider
from model.multi_task_nfm import Multi_Task_NFM
from transform.abortion_prediction_transform import AbortionPredictionTransformPipeline
from module.future_generate_main import FeatureGenerateMain

# 设置浮点数显示为小数点后2位，抑制科学计数法
# np.set_printoptions(precision=2, suppress=True)

# 设置随机种子
torch.manual_seed(config.RANDOM_SEED)
np.random.seed(config.RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(config.RANDOM_SEED)

# 初始化日志
logger = setup_logger(logger_config.TRAIN_LOG_FILE_PATH, logger_name="TrainLogger")

def _collate_fn(batch):
    # 初始化列表
    features_list = []
    
    # 多标签任务（前三个标签）
    multi_label_list = []
    
    for feature, label in batch:
        # 添加特征
        features_list.append(feature)
        
        # 多标签任务（前三个）- 需要作为一个向量
        multi_label = label
        multi_label_list.append(multi_label)
    
    # 转换为张量
    features_tensor = torch.tensor(np.array(features_list), dtype=torch.float32)
    multi_label_tensor = torch.tensor(np.array(multi_label_list), dtype=torch.float32)  # 多标签用float
    
    # 返回特征和所有标签
    return features_tensor, multi_label_tensor

    # 针对空值做处理
def mask_feature_null(data = None, mode='train'):
    if data is not None:
        logger.info("数据加载成功，开始处理特征空值掩码...")
    else:
        logger.error("数据加载失败，无法处理特征空值掩码。")

    # 1 有效 0无效
    logger.info("开始处理特征空值掩码...")
    for feature in ColumnsConfig.feature_columns:
        data[feature + '_mask'] = (1 - data[feature].isnull().astype(int))
    
    # 获取特征列和mask列的名称
    features = [col for col in data.columns if not col.endswith('_mask')]
    masks = [col for col in data.columns if col.endswith('_mask')]
    # 按特征名和mask特征名交替排序 确保每个特征后面紧跟它的掩码列
    sorted_columns = []
    for feature in features:
        sorted_columns.append(feature)
        mask_col = feature + '_mask'
        if mask_col in masks:
            sorted_columns.append(mask_col)
    # 重新排列DataFrame的列
    # train_X_transformed_Mask_Null = train_X_transformed[['date_code'] + self.sorted_columns]
    # test_X_transformed_Mask_Null = test_X_transformed[['date_code'] + self.sorted_columns]
    data_transformed_Mask_Null = data[sorted_columns]
    if mode == 'train':
        save_to_csv(filepath=DataPathConfig.DATA_TRANSFORMED_MASK_NULL_TRAIN_PATH, df=data_transformed_Mask_Null)
    elif mode == 'val':
        save_to_csv(filepath=DataPathConfig.DATA_TRANSFORMED_MASK_NULL_VAL_PATH, df=data_transformed_Mask_Null)
    logger.info("特征空值掩码处理完成，数据已保存。")
    return data_transformed_Mask_Null

def split_data(data_transformed_masked_null, y):
    
    periods = [(1, 7), (8, 14), (15, 21)]
    has_risk_label_list = [ColumnsConfig.HAS_RISK_4_CLASS_PRE.format(left, right) for left, right in periods]
    # 确保日期列是datetime类型
    data_transformed_masked_null = data_transformed_masked_null.copy()
    y = y.copy()

    train_X, test_X, train_y, test_y = train_test_split(data_transformed_masked_null, y, test_size=config.TEST_SPLIT_RATIO, random_state=config.RANDOM_SEED)

    train_X = train_X[ColumnsConfig.feature_columns]
    test_X = test_X[ColumnsConfig.feature_columns]
    train_y = train_y[has_risk_label_list]
    test_y = test_y[has_risk_label_list]

    return train_X, test_X, train_y, test_y


# 基于概率的随机采样计算
def sample_num_balance(X, y, random_state=42):
    """
    为每个标签创建单独的平衡数据集，每个数据集中对应的标签正负样本比例为1:1
    
    参数:
    - X: 特征矩阵
    - y: 标签DataFrame，包含三个标签列(abort_1_7, abort_8_14, abort_15_21)
    - random_state: 随机种子，确保结果可复现
    
    返回:
    - abort_1_7_balanced: 第一个标签平衡数据集
    - abort_8_14_balanced: 第二个标签平衡数据集
    - abort_15_21_balanced: 第三个标签平衡数据集
    """
    # 设置随机种子
    np.random.seed(random_state)
    
    # 合并特征和标签
    feature_label = pd.concat([X, y], axis=1)
    
    # 获取标签列
    label_columns = y.columns.tolist()
    
    # 为每个标签创建一个平衡数据集
    balanced_datasets = {}
    
    for label in label_columns:
        # 找出正样本和负样本
        positive_samples = feature_label[feature_label[label] == 1]
        negative_samples = feature_label[feature_label[label] == 0]
        
        # 确保有足够的正负样本
        if len(positive_samples) == 0 or len(negative_samples) == 0:
            print(f"警告：标签 {label} 缺少正样本或负样本，使用原始数据集")
            balanced_datasets[label] = feature_label.copy()
            continue
        
        # 计算采样数量
        sample_size = min(len(positive_samples), len(negative_samples))
        
        # 如果正样本数量大于采样数量，对正样本进行下采样
        if len(positive_samples) > sample_size:
            positive_samples = positive_samples.sample(n=sample_size, random_state=random_state)
        
        # 对负样本进行下采样
        negative_samples = negative_samples.sample(n=sample_size, random_state=random_state)
        
        # 合并正样本和采样的负样本
        balanced_label_dataset = pd.concat([positive_samples, negative_samples])
        
        # 随机打乱
        # balanced_label_dataset = shuffle(balanced_label_dataset, random_state=random_state)
        
        # 保存到字典
        balanced_datasets[label] = balanced_label_dataset
        
        # 打印平衡后的统计信息
        pos_count = sum(balanced_label_dataset[label] == 1)
        neg_count = sum(balanced_label_dataset[label] == 0)
        print(f"标签 {label} 平衡后: 正样本 {pos_count}, 负样本 {neg_count}, 比例 {pos_count/neg_count:.2f}")
    
    # 返回三个平衡数据集
    abort_1_7_balanced = balanced_datasets.get(label_columns[0], feature_label.copy())
    abort_8_14_balanced = balanced_datasets.get(label_columns[1], feature_label.copy())  
    abort_15_21_balanced = balanced_datasets.get(label_columns[2], feature_label.copy())

    abort_1_7_balanced.to_csv('abort_1_7_balanced.csv', index=False)
    abort_8_14_balanced.to_csv('abort_8_14_balanced.csv', index=False)
    abort_15_21_balanced.to_csv('abort_15_21_balanced.csv', index=False)

    abort_1_7_balanced_X = abort_1_7_balanced.drop(columns=label_columns)
    abort_8_14_balanced_X = abort_8_14_balanced.drop(columns=label_columns)
    abort_15_21_balanced_X = abort_15_21_balanced.drop(columns=label_columns)
    abort_1_7_balanced_y = abort_1_7_balanced[label_columns]
    abort_8_14_balanced_y = abort_8_14_balanced[label_columns]
    abort_15_21_balanced_y = abort_15_21_balanced[label_columns]
    
    return abort_1_7_balanced_X, abort_8_14_balanced_X, abort_15_21_balanced_X, abort_1_7_balanced_y, abort_8_14_balanced_y, abort_15_21_balanced_y

def sample_num_balance_same_date(X, y, random_state=42):
    """
    为每个标签创建单独的平衡数据集，每个数据集中对应的标签正负样本比例为1:1
    正样本会匹配与其同一天(stats_dt)的负样本
    
    参数:
    - X: 特征矩阵
    - y: 标签DataFrame，包含三个标签列(abort_1_7, abort_8_14, abort_15_21)
    - random_state: 随机种子，确保结果可复现
    
    返回:
    - abort_1_7_balanced_X, abort_8_14_balanced_X, abort_15_21_balanced_X: 三个标签的平衡特征
    - abort_1_7_balanced_y, abort_8_14_balanced_y, abort_15_21_balanced_y: 三个标签的平衡标签
    """
    # 设置随机种子
    np.random.seed(random_state)
    
    # 合并特征和标签
    feature_label = pd.concat([X, y], axis=1)
    
    # 获取标签列
    label_columns = y.columns.tolist()
    
    # 为每个标签创建一个平衡数据集
    balanced_datasets = {}
    
    for label in label_columns:
        # 按日期和标签值分组
        positive_samples = feature_label[feature_label[label] == 1]
        negative_samples = feature_label[feature_label[label] == 0]
        
        if len(positive_samples) == 0 or len(negative_samples) == 0:
            print(f"警告：标签 {label} 缺少正样本或负样本，使用原始数据集")
            balanced_datasets[label] = feature_label.copy()
            continue
        
        # 按日期分组的正样本
        positive_by_date = positive_samples.groupby('stats_dt')
        
        # 收集所有匹配的样本
        matched_samples = []
        unmatched_positives = []
        
        # 对每个日期的正样本进行处理
        for date, pos_group in positive_by_date:
            # 找到同一天的负样本
            same_day_negatives = negative_samples[negative_samples['stats_dt'] == date]
            
            if len(same_day_negatives) > 0:
                # 如果有足够的负样本，随机抽取与正样本数量相同的负样本
                # 如果负样本不足，则全部使用，这时会导致不平衡
                sample_size = min(len(pos_group), len(same_day_negatives))
                sampled_negatives = same_day_negatives.sample(n=sample_size, random_state=random_state)
                
                # 如果需要下采样正样本，确保平衡
                sampled_positives = pos_group
                if len(pos_group) > sample_size:
                    sampled_positives = pos_group.sample(n=sample_size, random_state=random_state)
                
                # 合并这一天的样本
                matched_samples.append(pd.concat([sampled_positives, sampled_negatives]))
            else:
                # 如果这一天没有负样本，记录这些正样本稍后处理
                unmatched_positives.append(pos_group)
        
        # 处理未匹配的正样本（可选）
        if unmatched_positives:
            unmatched_pos_df = pd.concat(unmatched_positives)
            print(f"标签 {label} 有 {len(unmatched_pos_df)} 个正样本没有同日期的负样本")
            
            # 策略1：从所有负样本中随机抽取
            remaining_negatives = negative_samples.copy()
            if len(remaining_negatives) >= len(unmatched_pos_df):
                # 随机抽取剩余的负样本
                sampled_negatives = remaining_negatives.sample(n=len(unmatched_pos_df), random_state=random_state)
                matched_samples.append(pd.concat([unmatched_pos_df, sampled_negatives]))
            else:
                print(f"警告：标签 {label} 没有足够的负样本进行匹配")
                # 使用所有剩余负样本
                matched_samples.append(pd.concat([unmatched_pos_df, remaining_negatives]))
        
        # 合并所有匹配的样本
        if matched_samples:
            balanced_label_dataset = pd.concat(matched_samples)
        else:
            print(f"警告：标签 {label} 没有匹配的样本，使用原始数据集")
            balanced_label_dataset = feature_label.copy()
        
        # 随机打乱
        balanced_label_dataset = balanced_label_dataset.sample(frac=1, random_state=random_state)
        
        # 保存到字典
        balanced_datasets[label] = balanced_label_dataset
        
        # 打印平衡后的统计信息
        pos_count = sum(balanced_label_dataset[label] == 1)
        neg_count = sum(balanced_label_dataset[label] == 0)
        print(f"标签 {label} 平衡后: 正样本 {pos_count}, 负样本 {neg_count}, 比例 {pos_count/neg_count:.2f}")
    
    # 返回三个平衡数据集
    abort_1_7_balanced = balanced_datasets.get(label_columns[0], feature_label.copy())
    abort_8_14_balanced = balanced_datasets.get(label_columns[1], feature_label.copy())  
    abort_15_21_balanced = balanced_datasets.get(label_columns[2], feature_label.copy())

    # 可选：保存到CSV文件
    abort_1_7_balanced.to_csv('abort_1_7_balanced.csv', index=False)
    abort_8_14_balanced.to_csv('abort_8_14_balanced.csv', index=False)
    abort_15_21_balanced.to_csv('abort_15_21_balanced.csv', index=False)

    # 分离特征和标签
    abort_1_7_balanced_X = abort_1_7_balanced.drop(columns=label_columns)
    abort_8_14_balanced_X = abort_8_14_balanced.drop(columns=label_columns)
    abort_15_21_balanced_X = abort_15_21_balanced.drop(columns=label_columns)
    abort_1_7_balanced_y = abort_1_7_balanced[label_columns]
    abort_8_14_balanced_y = abort_8_14_balanced[label_columns]
    abort_15_21_balanced_y = abort_15_21_balanced[label_columns]
    
    return abort_1_7_balanced_X, abort_8_14_balanced_X, abort_15_21_balanced_X, abort_1_7_balanced_y, abort_8_14_balanced_y, abort_15_21_balanced_y

def sample_num_balance_same_l3(X, y, random_state=42):
    """
    为每个标签创建单独的平衡数据集，每个数据集中对应的标签正负样本比例为1:1
    正样本会匹配与其同一天(stats_dt)的负样本
    
    参数:
    - X: 特征矩阵
    - y: 标签DataFrame，包含三个标签列(abort_1_7, abort_8_14, abort_15_21)
    - random_state: 随机种子，确保结果可复现
    
    返回:
    - abort_1_7_balanced_X, abort_8_14_balanced_X, abort_15_21_balanced_X: 三个标签的平衡特征
    - abort_1_7_balanced_y, abort_8_14_balanced_y, abort_15_21_balanced_y: 三个标签的平衡标签
    """
    # 设置随机种子
    np.random.seed(random_state)
    
    # 合并特征和标签
    feature_label = pd.concat([X, y], axis=1)
    
    # 获取标签列
    label_columns = y.columns.tolist()
    
    # 为每个标签创建一个平衡数据集
    balanced_datasets = {}
    
    for label in label_columns:
        # 按日期和标签值分组
        positive_samples = feature_label[feature_label[label] == 1]
        negative_samples = feature_label[feature_label[label] == 0]
        
        if len(positive_samples) == 0 or len(negative_samples) == 0:
            print(f"警告：标签 {label} 缺少正样本或负样本，使用原始数据集")
            balanced_datasets[label] = feature_label.copy()
            continue
        
        # 按日期分组的正样本
        positive_by_l3 = positive_samples.groupby('l3_org_inv_dk')
        
        # 收集所有匹配的样本
        matched_samples = []
        unmatched_positives = []
        
        # 对每个日期的正样本进行处理
        for date, pos_group in positive_by_l3:
            # 找到同一天的负样本
            same_l3_negatives = negative_samples[negative_samples['l3_org_inv_dk'] == date]
            
            if len(same_l3_negatives) > 0:
                # 如果有足够的负样本，随机抽取与正样本数量相同的负样本
                # 如果负样本不足，则全部使用，这时会导致不平衡
                sample_size = min(len(pos_group), len(same_l3_negatives))
                sampled_negatives = same_l3_negatives.sample(n=sample_size, random_state=random_state)
                
                # 如果需要下采样正样本，确保平衡
                sampled_positives = pos_group
                if len(pos_group) > sample_size:
                    sampled_positives = pos_group.sample(n=sample_size, random_state=random_state)
                
                # 合并这一天的样本
                matched_samples.append(pd.concat([sampled_positives, sampled_negatives]))
            else:
                # 如果这一天没有负样本，记录这些正样本稍后处理
                unmatched_positives.append(pos_group)
        
        # 处理未匹配的正样本（可选）
        if unmatched_positives:
            unmatched_pos_df = pd.concat(unmatched_positives)
            print(f"标签 {label} 有 {len(unmatched_pos_df)} 个正样本没有同日期的负样本")
            
            # 策略1：从所有负样本中随机抽取
            remaining_negatives = negative_samples.copy()
            if len(remaining_negatives) >= len(unmatched_pos_df):
                # 随机抽取剩余的负样本
                sampled_negatives = remaining_negatives.sample(n=len(unmatched_pos_df), random_state=random_state)
                matched_samples.append(pd.concat([unmatched_pos_df, sampled_negatives]))
            else:
                print(f"警告：标签 {label} 没有足够的负样本进行匹配")
                # 使用所有剩余负样本
                matched_samples.append(pd.concat([unmatched_pos_df, remaining_negatives]))
        
        # 合并所有匹配的样本
        if matched_samples:
            balanced_label_dataset = pd.concat(matched_samples)
        else:
            print(f"警告：标签 {label} 没有匹配的样本，使用原始数据集")
            balanced_label_dataset = feature_label.copy()
        
        # 随机打乱
        balanced_label_dataset = balanced_label_dataset.sample(frac=1, random_state=random_state)
        
        # 保存到字典
        balanced_datasets[label] = balanced_label_dataset
        
        # 打印平衡后的统计信息
        pos_count = sum(balanced_label_dataset[label] == 1)
        neg_count = sum(balanced_label_dataset[label] == 0)
        print(f"标签 {label} 平衡后: 正样本 {pos_count}, 负样本 {neg_count}, 比例 {pos_count/neg_count:.2f}")
    
    # 返回三个平衡数据集
    abort_1_7_balanced = balanced_datasets.get(label_columns[0], feature_label.copy())
    abort_8_14_balanced = balanced_datasets.get(label_columns[1], feature_label.copy())  
    abort_15_21_balanced = balanced_datasets.get(label_columns[2], feature_label.copy())

    # 可选：保存到CSV文件
    abort_1_7_balanced.to_csv('abort_1_7_balanced.csv', index=False)
    abort_8_14_balanced.to_csv('abort_8_14_balanced.csv', index=False)
    abort_15_21_balanced.to_csv('abort_15_21_balanced.csv', index=False)

    # 分离特征和标签
    abort_1_7_balanced_X = abort_1_7_balanced.drop(columns=label_columns)
    abort_8_14_balanced_X = abort_8_14_balanced.drop(columns=label_columns)
    abort_15_21_balanced_X = abort_15_21_balanced.drop(columns=label_columns)
    abort_1_7_balanced_y = abort_1_7_balanced[label_columns]
    abort_8_14_balanced_y = abort_8_14_balanced[label_columns]
    abort_15_21_balanced_y = abort_15_21_balanced[label_columns]
    
    return abort_1_7_balanced_X, abort_8_14_balanced_X, abort_15_21_balanced_X, abort_1_7_balanced_y, abort_8_14_balanced_y, abort_15_21_balanced_y

def sample_num_balance_same_l3_date(X, y, random_state=42):
    """
    为每个标签创建单独的平衡数据集，每个数据集中对应的标签正负样本比例为1:1
    正样本会匹配与其同一天(stats_dt)且同一三级单位(l3_org_inv_dk)的负样本
    
    参数:
    - X: 特征矩阵
    - y: 标签DataFrame，包含三个标签列(abort_1_7, abort_8_14, abort_15_21)
    - random_state: 随机种子，确保结果可复现
    
    返回:
    - abort_1_7_balanced_X, abort_8_14_balanced_X, abort_15_21_balanced_X: 三个标签的平衡特征
    - abort_1_7_balanced_y, abort_8_14_balanced_y, abort_15_21_balanced_y: 三个标签的平衡标签
    """
    # 设置随机种子
    np.random.seed(random_state)
    
    # 合并特征和标签
    feature_label = pd.concat([X, y], axis=1)
    
    # 获取标签列
    label_columns = y.columns.tolist()
    
    # 为每个标签创建一个平衡数据集
    balanced_datasets = {}
    
    for label in label_columns:
        # 获取正样本和负样本
        positive_samples = feature_label[feature_label[label] == 1]
        negative_samples = feature_label[feature_label[label] == 0]
        
        if len(positive_samples) == 0 or len(negative_samples) == 0:
            print(f"警告：标签 {label} 缺少正样本或负样本，使用原始数据集")
            balanced_datasets[label] = feature_label.copy()
            continue
        
        # 按日期和三级单位组合分组的正样本
        positive_by_date_l3 = positive_samples.groupby(['stats_dt', 'l3_org_inv_dk'])
        
        # 收集所有匹配的样本
        matched_samples = []
        unmatched_positives = []
        
        # 对每个日期和三级单位组合的正样本进行处理
        for (date, l3_org), pos_group in positive_by_date_l3:
            # 找到同一天且同一三级单位的负样本
            same_date_l3_negatives = negative_samples[
                (negative_samples['stats_dt'] == date) & 
                (negative_samples['l3_org_inv_dk'] == l3_org)
            ]
            
            if len(same_date_l3_negatives) > 0:
                # 如果有足够的负样本，随机抽取与正样本数量相同的负样本
                # 如果负样本不足，则全部使用，这时会导致不平衡
                sample_size = min(len(pos_group), len(same_date_l3_negatives))
                sampled_negatives = same_date_l3_negatives.sample(n=sample_size, random_state=random_state)
                
                # 如果需要下采样正样本，确保平衡
                sampled_positives = pos_group
                if len(pos_group) > sample_size:
                    sampled_positives = pos_group.sample(n=sample_size, random_state=random_state)
                
                # 合并这一组样本
                matched_samples.append(pd.concat([sampled_positives, sampled_negatives]))
            else:
                # 如果没有同日期同三级单位的负样本，记录这些正样本稍后处理
                unmatched_positives.append(pos_group)
        
        # 处理未匹配的正样本（分两步：先尝试匹配同日期，再尝试匹配同三级单位）
        if unmatched_positives:
            unmatched_pos_df = pd.concat(unmatched_positives)
            total_unmatched = len(unmatched_pos_df)
            print(f"标签 {label} 有 {total_unmatched} 个正样本没有同日期同三级单位的负样本")
            
            # 步骤1：尝试匹配同日期的负样本
            unmatched_by_date = unmatched_pos_df.groupby('stats_dt')
            still_unmatched = []
            
            for date, pos_date_group in unmatched_by_date:
                same_date_negatives = negative_samples[negative_samples['stats_dt'] == date]
                
                if len(same_date_negatives) > 0:
                    sample_size = min(len(pos_date_group), len(same_date_negatives))
                    sampled_negatives = same_date_negatives.sample(n=sample_size, random_state=random_state)
                    
                    if len(pos_date_group) > sample_size:
                        pos_date_group = pos_date_group.sample(n=sample_size, random_state=random_state)
                    
                    matched_samples.append(pd.concat([pos_date_group, sampled_negatives]))
                else:
                    still_unmatched.append(pos_date_group)
            
            # 步骤2：尝试匹配同三级单位的负样本
            if still_unmatched:
                still_unmatched_df = pd.concat(still_unmatched)
                unmatched_by_l3 = still_unmatched_df.groupby('l3_org_inv_dk')
                final_unmatched = []
                
                for l3_org, pos_l3_group in unmatched_by_l3:
                    same_l3_negatives = negative_samples[negative_samples['l3_org_inv_dk'] == l3_org]
                    
                    if len(same_l3_negatives) > 0:
                        sample_size = min(len(pos_l3_group), len(same_l3_negatives))
                        sampled_negatives = same_l3_negatives.sample(n=sample_size, random_state=random_state)
                        
                        if len(pos_l3_group) > sample_size:
                            pos_l3_group = pos_l3_group.sample(n=sample_size, random_state=random_state)
                        
                        matched_samples.append(pd.concat([pos_l3_group, sampled_negatives]))
                    else:
                        final_unmatched.append(pos_l3_group)
                
                # 步骤3：对剩余未匹配的使用随机负样本
                if final_unmatched:
                    final_unmatched_df = pd.concat(final_unmatched)
                    remaining_unmatched = len(final_unmatched_df)
                    print(f"标签 {label} 仍有 {remaining_unmatched} 个正样本无法匹配到同日期或同三级单位的负样本")
                    
                    if len(negative_samples) >= remaining_unmatched:
                        # 随机抽取负样本
                        sampled_negatives = negative_samples.sample(n=remaining_unmatched, random_state=random_state)
                        matched_samples.append(pd.concat([final_unmatched_df, sampled_negatives]))
                    else:
                        print(f"警告：标签 {label} 没有足够的负样本进行匹配")
                        # 使用全部剩余负样本
                        matched_samples.append(pd.concat([final_unmatched_df, negative_samples]))
        
        # 合并所有匹配的样本
        if matched_samples:
            balanced_label_dataset = pd.concat(matched_samples)
        else:
            print(f"警告：标签 {label} 没有匹配的样本，使用原始数据集")
            balanced_label_dataset = feature_label.copy()
        
        # 随机打乱
        balanced_label_dataset = balanced_label_dataset.sample(frac=1, random_state=random_state)
        
        # 保存到字典
        balanced_datasets[label] = balanced_label_dataset
        
        # 打印平衡后的统计信息
        pos_count = sum(balanced_label_dataset[label] == 1)
        neg_count = sum(balanced_label_dataset[label] == 0)
        print(f"标签 {label} 平衡后: 正样本 {pos_count}, 负样本 {neg_count}, 比例 {pos_count/neg_count:.2f}")
    
    # 返回三个平衡数据集
    abort_1_7_balanced = balanced_datasets.get(label_columns[0], feature_label.copy())
    abort_8_14_balanced = balanced_datasets.get(label_columns[1], feature_label.copy())  
    abort_15_21_balanced = balanced_datasets.get(label_columns[2], feature_label.copy())

    # 可选：保存到CSV文件
    abort_1_7_balanced.to_csv('abort_1_7_balanced.csv', index=False)
    abort_8_14_balanced.to_csv('abort_8_14_balanced.csv', index=False)
    abort_15_21_balanced.to_csv('abort_15_21_balanced.csv', index=False)

    # 分离特征和标签
    abort_1_7_balanced_X = abort_1_7_balanced.drop(columns=label_columns)
    abort_8_14_balanced_X = abort_8_14_balanced.drop(columns=label_columns)
    abort_15_21_balanced_X = abort_15_21_balanced.drop(columns=label_columns)
    abort_1_7_balanced_y = abort_1_7_balanced[label_columns]
    abort_8_14_balanced_y = abort_8_14_balanced[label_columns]
    abort_15_21_balanced_y = abort_15_21_balanced[label_columns]
    
    return abort_1_7_balanced_X, abort_8_14_balanced_X, abort_15_21_balanced_X, abort_1_7_balanced_y, abort_8_14_balanced_y, abort_15_21_balanced_y



def train_model(model, train_loaders, val_loaders, criterion, optimizer, num_epochs, device, early_stopping=None):
    logger.info("开始训练...")
    # scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.8, patience=1)
    
    abort_1_7_banlanced_train_lodaer, abort_8_14_banlanced_train_lodaer, abort_15_21_banlanced_train_lodaer = train_loaders
    abort_1_7_banlanced_val_lodaer, abort_8_14_banlanced_val_lodaer, abort_15_21_banlanced_val_lodaer = val_loaders

    label_names = ['abort_1_7', 'abort_8_14', 'abort_15_21']
    def cycle_iterator(iterable): # 当批次数最小的loader迭代完时，重新开始迭代
        """创建一个循环迭代器，当迭代完成时重新开始"""
        iterator = iter(iterable)
        while True:
            try:
                yield next(iterator)
            except StopIteration:
                iterator = iter(iterable)
                yield next(iterator)
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        batch_count = 0
        
        # 获取最长的数据集长度
        max_batches = max(len(loader) for loader in train_loaders)
        
        # 为每个dataloader创建循环迭代器
        iterators = [cycle_iterator(loader) for loader in train_loaders]
        
        # 训练max_batches次
        for _ in range(max_batches):
            optimizer.zero_grad()
            combined_loss = 0
            
            # 从每个loader获取一个batch
            for i, iterator in enumerate(iterators):
                batch_x, batch_y = next(iterator)
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                
                outputs = model(batch_x)
                # 获取当前任务的预测和标签
                current_pred = outputs[:, i]
                current_label = batch_y[:, i]  # 假设batch_y包含所有标签
                
                # 计算当前任务的损失
                loss = criterion(current_pred, current_label)
                combined_loss += loss
            
            # 计算平均损失（除以标签数量）
            combined_loss = combined_loss / len(iterators)  # 除以3
            # 反向传播总损失
            combined_loss.backward()
            optimizer.step()
            
            total_loss += combined_loss.item()
            batch_count += 1
        
        avg_train_loss = total_loss / batch_count

        # --- 验证阶段 ---
        avg_val_loss, metrics = evaluate_model(model, val_loaders, criterion, device)
        logger.info(f"Epoch [{epoch+1}/{num_epochs}], 训练集平均损失：{avg_train_loss:.4f}, 验证集平均损失: {avg_val_loss:.4f}")
        
        # --- 早停检查 ---
        if early_stopping is not None:
            # 使用验证集损失或其他指标作为早停依据，这里使用验证集F1分数
            early_stopping(avg_val_loss, model)
            
            if early_stopping.early_stop:
                logger.info(f"触发早停! 在Epoch {epoch+1}/{num_epochs}")
                break
    
    # 如果使用了早停，加载最佳模型
    if early_stopping is not None and os.path.exists(early_stopping.path):
        logger.info(f"加载最佳模型权重: {early_stopping.path}")
        model.load_state_dict(torch.load(early_stopping.path))
    
    # --- 最终评估 ---
    avg_train_loss, train_metrics = evaluate_model(model, train_loaders, criterion, device)
    avg_val_loss, val_metrics = evaluate_model(model, val_loaders, criterion, device)
    
    # --- 打印最终评估结果 ---
    logger.info(f"最终评估 - 训练集损失: {avg_train_loss:.4f}, 验证集损失: {avg_val_loss:.4f}")
    logger.info("训练完成.")

    return model

def evaluate_model(model, val_loaders, criterion, device):
    """
    评估模型在各个标签上的性能
    
    参数:
    - model: 训练好的多标签模型
    - val_loaders: 三个平衡验证集的数据加载器列表 [val_loader_1_7, val_loader_8_14, val_loader_15_21]
    - complete_val_loader: 完整验证集(可选)
    - criterion: 损失函数 (BCEWithLogitsLoss)
    - device: 计算设备
    """
    model.eval()
    label_names = ['abort_1_7', 'abort_8_14', 'abort_15_21']
    metrics = {}
    
    # 1. 在每个平衡验证集上分别评估对应的标签
    print("=" * 50)
    print("在平衡验证集上评估各标签性能:")
    
    total_loss = 0 # 累加所有标签的平均损失
    for i, (val_loader, label_name) in enumerate(zip(val_loaders, label_names)):
        task_loss = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                outputs = model(batch_x)
                
                # 只评估当前任务的标签
                current_pred = outputs[:, i]
                current_label = batch_y[:, i]
                
                loss = criterion(current_pred, current_label)
                task_loss += loss.item()
                
                # 收集预测和真实标签
                pred_probs = torch.sigmoid(current_pred).cpu().numpy()
                true_labels = current_label.cpu().numpy()
                
                all_preds.extend(pred_probs)
                all_labels.extend(true_labels)
        
        # 计算评估指标
        auc = roc_auc_score(all_labels, all_preds)
        binary_preds = [1 if p >= 0.5 else 0 for p in all_preds]
        precision = precision_score(all_labels, binary_preds)
        recall = recall_score(all_labels, binary_preds)
        accuracy = accuracy_score(all_labels, binary_preds)
        
        metrics[label_name] = {
            'loss': task_loss/len(val_loader),
            'auc': auc,
            'precision': precision,
            'recall': recall,
            'accuracy': accuracy
        }
        
        total_loss += task_loss /  len(val_loader) # 累加当前标签的平均损失
        print(f"{label_name} - 损失: {task_loss/len(val_loader):.4f}, AUC: {auc:.4f}, "
              f"精确率: {precision:.4f}, 召回率: {recall:.4f}, 准确率: {accuracy:.4f}")
    
    avg_loss = total_loss / len(label_names) # 计算所有标签的整体平均损失
    return avg_loss, metrics

if __name__ == "__main__":
    logger.info("开始数据加载和预处理...")
    feature_gen_main = FeatureGenerateMain(
        running_dt=config.TRAIN_RUNNING_DT,
        origin_feature_precompute_interval=config.TRAIN_INTERVAL,
        logger=logger
    )
    feature_gen_main.generate_feature()  # 生成特征

    index_sample_obj = RiskPredictionIndexSampleDataset()
    connect_feature_obj = RiskPredictionFeatureDataset()
    logger.info('----------Create index sample---------')
    train_index_data = index_sample_obj.build_train_dataset(config.TRAIN_RUNNING_DT, config.TRAIN_INTERVAL)
    logger.info("----------Generating train dataset----------")
    train_connect_feature_data = connect_feature_obj.build_train_dataset(input_dataset=train_index_data.copy(), param=None)

    # 1. 加载和基础预处理数据
    # 生成特征
    # feature_generator = FeatureGenerator(running_dt=config.TRAIN_RUNNING_DT, interval_days=config.TRAIN_INTERVAL)
    # feature_df = feature_generator.generate_features()

    # 生成lable
    label_generator = LabelGenerator(
        feature_data=train_connect_feature_data,
        running_dt=config.TRAIN_RUNNING_DT,
        interval_days=config.TRAIN_INTERVAL
    )
    logger.info("开始生成标签...")
    X, y = label_generator.has_risk_period_generate_multi_label_alter_nodays()
    logger.info(f"标签计算完成，特征字段为：{X.columns}， 标签数据字段为：{y.columns}")
    logger.info(f"X,y特征数据形状为：{X.shape}， 标签数据形状为：{y.shape}")
    
    abort_1_7_balanced_X, abort_8_14_balanced_X, abort_15_21_balanced_X, abort_1_7_balanced_y, abort_8_14_balanced_y, abort_15_21_balanced_y = sample_num_balance_same_l3_date(X=X, y=y, random_state=config.RANDOM_SEED)  # 平衡样本数量
    abort_1_7_balanced_train_X, abort_1_7_balanced_val_X, abort_1_7_balanced_train_y, abort_1_7_balanced_val_y= split_data(abort_1_7_balanced_X, abort_1_7_balanced_y)
    abort_8_14_balanced_train_X, abort_8_14_balanced_val_X, abort_8_14_balanced_train_y, abort_8_14_balanced_val_y = split_data(abort_8_14_balanced_X, abort_8_14_balanced_y)
    abort_15_21_balanced_train_X, abort_15_21_balanced_val_X, abort_15_21_balanced_train_y, abort_15_21_balanced_val_y = split_data(abort_15_21_balanced_X, abort_15_21_balanced_y)
    
    abort_1_7_balanced_train_X.reset_index(drop=True, inplace=True)
    abort_8_14_balanced_train_X.reset_index(drop=True, inplace=True)
    abort_15_21_balanced_train_X.reset_index(drop=True, inplace=True)
    abort_1_7_balanced_train_y.reset_index(drop=True, inplace=True)
    abort_8_14_balanced_train_y.reset_index(drop=True, inplace=True)
    abort_15_21_balanced_train_y.reset_index(drop=True, inplace=True)

    abort_1_7_balanced_val_X.reset_index(drop=True, inplace=True)
    abort_8_14_balanced_val_X.reset_index(drop=True, inplace=True)
    abort_15_21_balanced_val_X.reset_index(drop=True, inplace=True)
    abort_1_7_balanced_val_y.reset_index(drop=True, inplace=True)
    abort_8_14_balanced_val_y.reset_index(drop=True, inplace=True)
    abort_15_21_balanced_val_y.reset_index(drop=True, inplace=True)

    logger.info(f"train_X数据字段为：{abort_1_7_balanced_train_X.columns}")
    logger.info(f"val_X数据字段为：{abort_1_7_balanced_train_y.columns}")
    logger.info(f"平衡标签比例后X数据形状为：{abort_1_7_balanced_train_X.shape}， 平衡标签比例后y数据形状为：{abort_1_7_balanced_train_y.shape}")

    total2transform = pd.concat([abort_1_7_balanced_train_X, abort_8_14_balanced_train_X, abort_15_21_balanced_train_X], axis=0, ignore_index=True) # 合并所有训练集特征数据

    param = {}
    transform = AbortionPredictionTransformPipeline(transform_feature_names = ColumnsConfig.feature_columns) # 传入使用特征

    # 对数据集进行处理
    total_transform = transform.fit_transform(input_dataset=total2transform) # 离散数据与连续数据处理

    abort_1_7_balanced_train_X_transformed = transform.transform(input_dataset=abort_1_7_balanced_train_X) # 离散数据与连续数据处理
    abort_8_14_balanced_train_X_transformed = transform.transform(input_dataset=abort_8_14_balanced_train_X) # 离散数据与连续数据处理
    abort_15_21_balanced_train_X_transformed = transform.transform(input_dataset=abort_15_21_balanced_train_X) # 离散数据与连续数据处理
    abort_1_7_balanced_val_X_transformed = transform.transform(input_dataset=abort_1_7_balanced_val_X) # 离散数据与连续数据处理
    abort_8_14_balanced_val_X_transformed = transform.transform(input_dataset=abort_8_14_balanced_val_X) # 离散数据与连续数据处理
    abort_15_21_balanced_val_X_transformed = transform.transform(input_dataset=abort_15_21_balanced_val_X) # 离散数据与连续数据处理

    abort_1_7_balanced_train_X_transformed.to_csv(
        DataPathConfig.TRAIN_TRANSFORMED_FEATURE_DATA_SAVE_PATH,
        index=False,
        encoding='utf-8-sig'
    )
    abort_1_7_balanced_val_X_transformed.to_csv(
        DataPathConfig.VAL_TRANSFORMED_FEATURE_DATA_SAVE_PATH,
        index=False,
        encoding='utf-8-sig'
    )

    with open(config.TRANSFORMER_SAVE_PATH, "w+") as dump_file:
        dump_file.write(transform.to_json()) # 保存为json 确保预测时使用与训练时相同的转换参数 比如使用测试集预测时，城市广州在训练时对应的id为2，预测时也为2
    logger.info("transformer transfrom_columns size: %d" % len(transform.features))
    logger.info("Saved transformer to {}".format(config.TRANSFORMER_SAVE_PATH))
    
    # 生成mask列
    abort_1_7_balanced_train_X_transformed_mask_null = mask_feature_null(data=abort_1_7_balanced_train_X_transformed, mode='train')
    abort_8_14_balanced_train_X_transformed_mask_null = mask_feature_null(data=abort_8_14_balanced_train_X_transformed, mode='train')
    abort_15_21_balanced_train_X_transformed_mask_null = mask_feature_null(data=abort_15_21_balanced_train_X_transformed, mode='train')
    abort_1_7_balanced_val_X_transformed_mask_null = mask_feature_null(data=abort_1_7_balanced_val_X_transformed, mode='val')
    abort_8_14_balanced_val_X_transformed_mask_null = mask_feature_null(data=abort_8_14_balanced_val_X_transformed, mode='val')
    abort_15_21_balanced_val_X_transformed_mask_null = mask_feature_null(data=abort_15_21_balanced_val_X_transformed, mode='val')

    abort_1_7_balanced_train_X_transformed_mask_null.fillna(0, inplace=True)  # 填充空值为0
    abort_8_14_balanced_train_X_transformed_mask_null.fillna(0, inplace=True)  # 填充空值为0
    abort_15_21_balanced_train_X_transformed_mask_null.fillna(0, inplace=True)  # 填充空值为0
    abort_1_7_balanced_val_X_transformed_mask_null.fillna(0, inplace=True)  # 填充空值为0
    abort_8_14_balanced_val_X_transformed_mask_null.fillna(0, inplace=True)  # 填充空值为0
    abort_15_21_balanced_val_X_transformed_mask_null.fillna(0, inplace=True)  # 填充空值为0

    logger.info(f"data_transformed_masked_null数据字段为：{abort_1_7_balanced_train_X_transformed_mask_null.columns}")

    abort_1_7_banlanced_train_df = pd.concat([abort_1_7_balanced_train_X_transformed_mask_null, abort_1_7_balanced_train_y], axis=1)
    abort_8_14_banlanced_train_df = pd.concat([abort_8_14_balanced_train_X_transformed_mask_null, abort_8_14_balanced_train_y], axis=1)
    abort_15_21_banlanced_train_df = pd.concat([abort_15_21_balanced_train_X_transformed_mask_null, abort_15_21_balanced_train_y], axis=1)
    abort_1_7_banlanced_val_df = pd.concat([abort_1_7_balanced_val_X_transformed_mask_null, abort_1_7_balanced_val_y], axis=1)
    abort_8_14_banlanced_val_df = pd.concat([abort_8_14_balanced_val_X_transformed_mask_null, abort_8_14_balanced_val_y], axis=1)
    abort_15_21_banlanced_val_df = pd.concat([abort_15_21_balanced_val_X_transformed_mask_null, abort_15_21_balanced_val_y], axis=1)

    abort_1_7_banlanced_train_df.reset_index(drop=True, inplace=True)
    abort_8_14_banlanced_train_df.reset_index(drop=True, inplace=True)
    abort_15_21_banlanced_train_df.reset_index(drop=True, inplace=True)
    abort_1_7_banlanced_val_df.reset_index(drop=True, inplace=True)
    abort_8_14_banlanced_val_df.reset_index(drop=True, inplace=True)
    abort_15_21_banlanced_val_df.reset_index(drop=True, inplace=True)

    logger.info(f"train_df数据字段为：{abort_1_7_banlanced_train_df.columns}")
    logger.info(f"val_df数据字段为：{abort_1_7_banlanced_val_df.columns}")
    
    abort_1_7_banlanced_train_df.to_csv(
        "train_df.csv",
        index=False,
        encoding='utf-8-sig'
    )

    # 5. 创建 PyTorch Dataset 和 DataLoader
    periods = [(1, 7), (8, 14), (15, 21)]
    has_risk_label_list = [ColumnsConfig.HAS_RISK_4_CLASS_PRE.format(left, right) for left, right in periods]

    abort_1_7_banlanced_train_dataset = MultiTaskAndMultiLabelDataset(abort_1_7_banlanced_train_df, label=has_risk_label_list)
    abort_8_14_banlanced_train_dataset = MultiTaskAndMultiLabelDataset(abort_8_14_banlanced_train_df, label=has_risk_label_list)
    abort_15_21_banlanced_train_dataset = MultiTaskAndMultiLabelDataset(abort_15_21_banlanced_train_df, label=has_risk_label_list)
    abort_1_7_banlanced_val_dataset = MultiTaskAndMultiLabelDataset(abort_1_7_banlanced_val_df, label=has_risk_label_list)
    abort_8_14_banlanced_val_dataset = MultiTaskAndMultiLabelDataset(abort_8_14_banlanced_val_df, label=has_risk_label_list)
    abort_15_21_banlanced_val_dataset = MultiTaskAndMultiLabelDataset(abort_15_21_banlanced_val_df, label=has_risk_label_list)

    abort_1_7_banlanced_train_lodaer = DataLoader(abort_1_7_banlanced_train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, collate_fn=_collate_fn, num_workers=config.NUM_WORKERS) # Windows下 num_workers>0 可能有问题
    abort_8_14_banlanced_train_lodaer = DataLoader(abort_8_14_banlanced_train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, collate_fn=_collate_fn, num_workers=config.NUM_WORKERS)
    abort_15_21_banlanced_train_lodaer = DataLoader(abort_15_21_banlanced_train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, collate_fn=_collate_fn, num_workers=config.NUM_WORKERS)
    abort_1_7_banlanced_val_lodaer = DataLoader(abort_1_7_banlanced_val_dataset, batch_size=config.BATCH_SIZE, shuffle=False, collate_fn=_collate_fn, num_workers=config.NUM_WORKERS)
    abort_8_14_banlanced_val_lodaer = DataLoader(abort_8_14_banlanced_val_dataset, batch_size=config.BATCH_SIZE, shuffle=False, collate_fn=_collate_fn, num_workers=config.NUM_WORKERS)
    abort_15_21_banlanced_val_lodaer = DataLoader(abort_15_21_banlanced_val_dataset, batch_size=config.BATCH_SIZE, shuffle=False, collate_fn=_collate_fn, num_workers=config.NUM_WORKERS)

    train_loaders = (abort_1_7_banlanced_train_lodaer, abort_8_14_banlanced_train_lodaer, abort_15_21_banlanced_train_lodaer)
    val_loaders = (abort_1_7_banlanced_val_lodaer, abort_8_14_banlanced_val_lodaer, abort_15_21_banlanced_val_lodaer)
    logger.info("数据加载器准备完毕.")

    # --- 模型、损失函数、优化器 ---
    feature_dict = transform.features.features
    Categorical_feature = ColumnsConfig.DISCRETE_COLUMNS # 离散值字段
    params = {
        'model_discrete_columns': ColumnsConfig.MODEL_DISCRETE_COLUMNS,
        'model_continuous_columns': ColumnsConfig.MODEL_CONTINUOUS_COLUMNS,
        'dropout': config.DROPOUT,

        'pigfarm_dk': feature_dict[Categorical_feature[0]].category_encode.size,
        'city': feature_dict[Categorical_feature[1]].category_encode.size,
        'season': 4,
    }
    model = Has_Risk_NFM_MultiLabel_7d1Linear(params).to(config.DEVICE) # 等待模型实现
    logger.info("模型初始化完成.")
    logger.info(f"模型结构:\n{model}")

    criterion = nn.BCEWithLogitsLoss()  # 假设是回归任务，使用均方误差
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)  # L2正则化

    # 初始化早停器
    early_stopping = EarlyStopping(
        patience=5,           # 连续5个epoch没有改善就停止
        verbose=True,         # 打印早停信息
        delta=0.001,          # 判定为改善的最小变化量
        path=config.MODEL_SAVE_PATH,  # 最佳模型保存路径
        trace_func=logger.info  # 使用logger记录信息
    )

    # --- 开始训练 (当前被注释掉，因为模型未定义) ---
    trained_model = train_model(model, train_loaders, val_loaders, criterion, optimizer, config.NUM_EPOCHS, config.DEVICE, early_stopping=early_stopping)


    # --- 模型评估 (可选，在测试集上) ---
    # ...

    # --- 保存最终模型 (如果未使用早停保存最佳模型) ---
    # final_model_path = os.path.join(config.MODEL_SAVE_PATH, "final_lstm_model.pt")
    # torch.save(trained_model.state_dict(), final_model_path)
    # logger.info(f"最终模型已保存至: {final_model_path}")

