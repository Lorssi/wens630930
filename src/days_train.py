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
import torch.nn.functional as F

# 从项目中导入模块
from configs.logger_config import logger_config
from configs.feature_config import ColumnsConfig, DataPathConfig
import config
from dataset.dataset import HasRiskDataset, MultiTaskDataset, DaysDataset
from utils.logger import setup_logger
from utils.early_stopping import EarlyStopping
from utils.save_csv import save_to_csv, read_csv
from feature.gen_feature import FeatureGenerator
from feature.gen_label import LabelGenerator
from transform.transform import FeatureTransformer
from model.mlp import Has_Risk_MLP
from model.nfm import Has_Risk_NFM
from model.days_nfm import Days_NFM
from transform.abortion_prediction_transform import AbortionPredictionTransformPipeline
from dataset.days_prediction_index_sample_dataset import DaysPredictionIndexSampleDataset
from dataset.days_prediction_feature_dataset import DaysPredictionFeatureDataset
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

class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        """
        Focal Loss 实现
        
        Args:
            alpha: 类别权重，可以是一个列表，为每个类别指定权重，处理类别不平衡
            gamma: 聚焦参数，提高对困难样本的关注度
            reduction: 损失计算方式，'mean'、'sum'或'none'
        """
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.reduction = reduction
        self.alpha = alpha
        if alpha is not None:
            if isinstance(alpha, list):
                self.alpha = torch.tensor(alpha)
            self.alpha = self.alpha.to(config.DEVICE)
    
    def forward(self, input, target):
        # 计算交叉熵损失
        ce_loss = F.cross_entropy(input, target, reduction='none', weight=self.alpha)
        
        # 获取目标类的概率
        pt = torch.exp(-ce_loss)
        
        # 计算Focal Loss
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        
        # 根据reduction方式处理损失
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

def calculate_class_weights(y_data, num_classes=8, method='inverse'):
    """
    计算类别权重，用于处理类别不平衡问题
    
    Args:
        y_data: 训练数据的标签，可以是一个pandas DataFrame或Series或numpy数组
        num_classes: 类别总数
        method: 权重计算方法，可选'inverse'(频率倒数),'log'(对数加权),'effective'(有效数量)
        
    Returns:
        torch.Tensor: 每个类别的权重
    """
    # 确保y_data是numpy数组
    if isinstance(y_data, pd.DataFrame) or isinstance(y_data, pd.Series):
        y_data = y_data.values

    # 将浮点标签转换为整数 - 添加这一行解决错误
    y_data = y_data.astype(np.int64)
    
    # 计算每个类别的样本数量
    class_counts = np.bincount(y_data.flatten(), minlength=num_classes)
    
    # 处理零样本情况，避免除零错误
    class_counts = np.where(class_counts == 0, 1, class_counts)
    
    if method == 'inverse':
        # 频率倒数: 样本越少，权重越大
        weights = 1.0 / class_counts
        # 归一化权重（使总和为num_classes）
        weights = weights * (num_classes / weights.sum())
    
    elif method == 'log':
        # 对数权重: 减轻极端不平衡的影响
        N = y_data.size
        weights = np.log(N / class_counts)
        # 处理可能的零或负值
        weights = np.where(weights <= 0, 1e-6, weights)
    
    elif method == 'effective':
        # 有效样本数: 介于倒数和对数之间的折中方案
        weights = 1.0 / np.sqrt(class_counts)
        # 归一化
        weights = weights * (num_classes / weights.sum())
    
    else:
        raise ValueError(f"不支持的计算方法: {method}")
    
    logger.info(f"计算的类别权重 ({method}): {weights}")
    
    return torch.tensor(weights, dtype=torch.float32)

# 计算每个任务的类别权重
def calculate_all_task_weights(train_y, num_classes=8, method='inverse'):
    """
    为所有三个任务计算类别权重
    
    Args:
        train_y: 包含三个任务标签的DataFrame
        num_classes: 每个任务的类别数
        method: 权重计算方法
        
    Returns:
        dict: 包含三个任务权重的字典
    """
    periods = [(1, 7), (8, 14), (15, 21)]
    days_label_list = [ColumnsConfig.DAYS_RISK_8_CLASS_PRE.format(start, end) for start, end in periods]
    
    weights = {}
    for i, label_col in enumerate(days_label_list):
        task_name = f"task_{periods[i][0]}_{periods[i][1]}"
        task_weights = calculate_class_weights(train_y[label_col], num_classes, method)
        weights[task_name] = task_weights
        logger.info(f"{task_name} 类别分布: {np.bincount(train_y[label_col], minlength=num_classes)}")
        
    return weights

def _collate_fn(batch):
    # 初始化列表
    features_list = []
    
    # 多任务分类（后三个标签）
    days_label_1_7_list = []
    days_label_8_14_list = []
    days_label_15_21_list = []
    
    for feature, label in batch:
        # 添加特征
        features_list.append(feature)
        
        # 多任务分类（后三个）- 每个任务单独处理
        days_label_1_7_list.append(label[0])  # 1-7天标签
        days_label_8_14_list.append(label[1]) # 8-14天标签
        days_label_15_21_list.append(label[2]) # 15-21天标签
    
    # 转换为张量
    features_tensor = torch.tensor(np.array(features_list), dtype=torch.float32)
    days_1_7_tensor = torch.tensor(np.array(days_label_1_7_list), dtype=torch.long)    # 分类标签用long
    days_8_14_tensor = torch.tensor(np.array(days_label_8_14_list), dtype=torch.long)
    days_15_21_tensor = torch.tensor(np.array(days_label_15_21_list), dtype=torch.long)
    
    # 返回特征和所有标签
    return features_tensor, days_1_7_tensor, days_8_14_tensor, days_15_21_tensor

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
    """
    按日期百分比划分训练集和验证集
    
    Args:
        data (DataFrame): 包含日期列的数据
        test_split_ratio (float): 验证集占比，默认0.2
        
    Returns:
        tuple: (训练集数据, 验证集数据)
    """
    # 确保日期列是datetime类型
    data_transformed_masked_null = data_transformed_masked_null.copy()
    y = y.copy()

    train_X, test_X, train_y, test_y = train_test_split(data_transformed_masked_null, y, test_size=config.TEST_SPLIT_RATIO, random_state=config.RANDOM_SEED)

    train_X = train_X[ColumnsConfig.feature_columns]
    test_X = test_X[ColumnsConfig.feature_columns]

    periods = [(1, 7), (8, 14), (15, 21)]
    days_label_list = [ColumnsConfig.DAYS_RISK_8_CLASS_PRE.format(start, end) for start, end in periods]
    train_y = train_y[days_label_list]
    test_y = test_y[days_label_list]

    return train_X, test_X, train_y, test_y

def train_model(model, train_loader, val_loader, criterions, optimizer, num_epochs, device, early_stopping=None):
    logger.info("开始训练...")
    # 解包三个损失函数
    criterion_1_7, criterion_8_14, criterion_15_21 = criterions
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for batch_idx, (features, days_1_7, days_8_14, days_15_21) in enumerate(train_loader):
            features, days_1_7, days_8_14, days_15_21 = features.to(device), days_1_7.to(device), days_8_14.to(device), days_15_21.to(device)

            optimizer.zero_grad()
            days_1_7_output, days_8_14_output, days_15_21_output = model(features)
            days_1_7_loss = criterion_1_7(days_1_7_output, days_1_7)
            days_8_14_loss = criterion_8_14(days_8_14_output, days_8_14)
            days_15_21_loss = criterion_15_21(days_15_21_output, days_15_21)

            days_loss = (days_1_7_loss + days_8_14_loss + days_15_21_loss) / 3
            loss = days_loss
            loss.backward()

            optimizer.step()
            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)

        # --- 验证阶段 ---
        avg_val_loss, metrics = eval(model, val_loader, criterions, device)
        
        # 打印训练指标 - 显示平均指标
        logger.info(f"Epoch [{epoch+1}/{num_epochs}], 训练集平均损失：{avg_train_loss:.4f}, 验证集平均损失: {avg_val_loss:.4f}")
        logger.info(f"Epoch [{epoch+1}/{num_epochs}], 平均指标 - 准确率: {metrics['accuracy']:.4f}, 精确率: {metrics['precision']:.4f}, 召回率: {metrics['recall']:.4f}, F1: {metrics['f1']:.4f}, AUC: {metrics['auc']:.4f}")
        
        # 打印各任务详细指标 - 显示所有指标
        logger.info(f"  任务1-7天 - 准确率: {metrics['task_1_7']['accuracy']:.4f}, 精确率: {metrics['task_1_7']['precision']:.4f}, 召回率: {metrics['task_1_7']['recall']:.4f}, F1: {metrics['task_1_7']['f1']:.4f}, AUC: {metrics['task_1_7']['auc']:.4f}")
        logger.info(f"  任务8-14天 - 准确率: {metrics['task_8_14']['accuracy']:.4f}, 精确率: {metrics['task_8_14']['precision']:.4f}, 召回率: {metrics['task_8_14']['recall']:.4f}, F1: {metrics['task_8_14']['f1']:.4f}, AUC: {metrics['task_8_14']['auc']:.4f}")
        logger.info(f"  任务15-21天 - 准确率: {metrics['task_15_21']['accuracy']:.4f}, 精确率: {metrics['task_15_21']['precision']:.4f}, 召回率: {metrics['task_15_21']['recall']:.4f}, F1: {metrics['task_15_21']['f1']:.4f}, AUC: {metrics['task_15_21']['auc']:.4f}")
        
        # --- 早停检查 ---
        if early_stopping is not None:
            early_stopping(avg_val_loss, model)
            
            if early_stopping.early_stop:
                logger.info(f"触发早停! 在Epoch {epoch+1}/{num_epochs}")
                break
    
    # 如果使用了早停，加载最佳模型
    if early_stopping is not None and os.path.exists(early_stopping.path):
        logger.info(f"加载最佳模型权重: {early_stopping.path}")
        model.load_state_dict(torch.load(early_stopping.path))
    
    # --- 最终评估 ---
    avg_train_loss, train_metrics = eval(model, train_loader, criterions, device)
    avg_val_loss, val_metrics = eval(model, val_loader, criterions, device)
    
    # --- 打印最终评估结果（详细版本）---
    logger.info(f"最终评估 - 训练集损失: {avg_train_loss:.4f}, 验证集损失: {avg_val_loss:.4f}")
    logger.info(f"最终评估 - 训练集平均: 准确率: {train_metrics['accuracy']:.4f}, 精确率: {train_metrics['precision']:.4f}, 召回率: {train_metrics['recall']:.4f}, F1: {train_metrics['f1']:.4f}, AUC: {train_metrics['auc']:.4f}")
    logger.info(f"最终评估 - 验证集平均: 准确率: {val_metrics['accuracy']:.4f}, 精确率: {val_metrics['precision']:.4f}, 召回率: {val_metrics['recall']:.4f}, F1: {val_metrics['f1']:.4f}, AUC: {val_metrics['auc']:.4f}")
    
    # 最终评估各任务详细指标
    logger.info("=== 最终评估 - 各任务详细指标 ===")
    logger.info(f"训练集 1-7天任务: 准确率: {train_metrics['task_1_7']['accuracy']:.4f}, 精确率: {train_metrics['task_1_7']['precision']:.4f}, 召回率: {train_metrics['task_1_7']['recall']:.4f}, F1: {train_metrics['task_1_7']['f1']:.4f}, AUC: {train_metrics['task_1_7']['auc']:.4f}")
    logger.info(f"训练集 8-14天任务: 准确率: {train_metrics['task_8_14']['accuracy']:.4f}, 精确率: {train_metrics['task_8_14']['precision']:.4f}, 召回率: {train_metrics['task_8_14']['recall']:.4f}, F1: {train_metrics['task_8_14']['f1']:.4f}, AUC: {train_metrics['task_8_14']['auc']:.4f}")
    logger.info(f"训练集 15-21天任务: 准确率: {train_metrics['task_15_21']['accuracy']:.4f}, 精确率: {train_metrics['task_15_21']['precision']:.4f}, 召回率: {train_metrics['task_15_21']['recall']:.4f}, F1: {train_metrics['task_15_21']['f1']:.4f}, AUC: {train_metrics['task_15_21']['auc']:.4f}")
    
    logger.info(f"验证集 1-7天任务: 准确率: {val_metrics['task_1_7']['accuracy']:.4f}, 精确率: {val_metrics['task_1_7']['precision']:.4f}, 召回率: {val_metrics['task_1_7']['recall']:.4f}, F1: {val_metrics['task_1_7']['f1']:.4f}, AUC: {val_metrics['task_1_7']['auc']:.4f}")
    logger.info(f"验证集 8-14天任务: 准确率: {val_metrics['task_8_14']['accuracy']:.4f}, 精确率: {val_metrics['task_8_14']['precision']:.4f}, 召回率: {val_metrics['task_8_14']['recall']:.4f}, F1: {val_metrics['task_8_14']['f1']:.4f}, AUC: {val_metrics['task_8_14']['auc']:.4f}")
    logger.info(f"验证集 15-21天任务: 准确率: {val_metrics['task_15_21']['accuracy']:.4f}, 精确率: {val_metrics['task_15_21']['precision']:.4f}, 召回率: {val_metrics['task_15_21']['recall']:.4f}, F1: {val_metrics['task_15_21']['f1']:.4f}, AUC: {val_metrics['task_15_21']['auc']:.4f}")
    
    logger.info("训练完成.")
    return model

def eval(model, val_loader, criterions, device):
    """
    评估模型在验证集上的性能（多任务版本 - 评估三个时段的预测结果）
    """
    criterion_1_7, criterion_8_14, criterion_15_21 = criterions

    model.eval()
    val_loss = 0.0
    
    # 为每个任务分别收集预测结果
    all_preds_1_7 = []
    all_targets_1_7 = []
    all_probs_1_7 = []
    
    all_preds_8_14 = []
    all_targets_8_14 = []
    all_probs_8_14 = []
    
    all_preds_15_21 = []
    all_targets_15_21 = []
    all_probs_15_21 = []
    
    with torch.no_grad():
        for features, days_1_7, days_8_14, days_15_21 in val_loader:
            features = features.to(device)
            days_1_7 = days_1_7.to(device)
            days_8_14 = days_8_14.to(device)
            days_15_21 = days_15_21.to(device)
            
            # 前向传播
            days_1_7_output, days_8_14_output, days_15_21_output = model(features)
            
            # 计算损失
            days_1_7_loss = criterion_1_7(days_1_7_output, days_1_7)
            days_8_14_loss = criterion_8_14(days_8_14_output, days_8_14)
            days_15_21_loss = criterion_15_21(days_15_21_output, days_15_21)
            total_loss = (days_1_7_loss + days_8_14_loss + days_15_21_loss) / 3
            val_loss += total_loss.item()
            
            # 获取预测结果 - 1-7天任务
            probs_1_7 = torch.softmax(days_1_7_output, dim=1)
            preds_1_7 = torch.argmax(days_1_7_output, dim=1)
            all_preds_1_7.extend(preds_1_7.cpu().numpy())
            all_targets_1_7.extend(days_1_7.cpu().numpy())
            all_probs_1_7.extend(probs_1_7.cpu().numpy())
            
            # 获取预测结果 - 8-14天任务
            probs_8_14 = torch.softmax(days_8_14_output, dim=1)
            preds_8_14 = torch.argmax(days_8_14_output, dim=1)
            all_preds_8_14.extend(preds_8_14.cpu().numpy())
            all_targets_8_14.extend(days_8_14.cpu().numpy())
            all_probs_8_14.extend(probs_8_14.cpu().numpy())
            
            # 获取预测结果 - 15-21天任务
            probs_15_21 = torch.softmax(days_15_21_output, dim=1)
            preds_15_21 = torch.argmax(days_15_21_output, dim=1)
            all_preds_15_21.extend(preds_15_21.cpu().numpy())
            all_targets_15_21.extend(days_15_21.cpu().numpy())
            all_probs_15_21.extend(probs_15_21.cpu().numpy())
    
    # 计算平均损失
    avg_val_loss = val_loss / len(val_loader)
    
    # 转换为numpy数组
    all_preds_1_7 = np.array(all_preds_1_7)
    all_targets_1_7 = np.array(all_targets_1_7)
    all_probs_1_7 = np.array(all_probs_1_7)
    
    all_preds_8_14 = np.array(all_preds_8_14)
    all_targets_8_14 = np.array(all_targets_8_14)
    all_probs_8_14 = np.array(all_probs_8_14)
    
    all_preds_15_21 = np.array(all_preds_15_21)
    all_targets_15_21 = np.array(all_targets_15_21)
    all_probs_15_21 = np.array(all_probs_15_21)
    
    # 计算每个任务的指标
    def calculate_task_metrics(preds, targets, probs, task_name):
        accuracy = accuracy_score(targets, preds)
        precision = precision_score(targets, preds, average='macro', zero_division=0)
        recall = recall_score(targets, preds, average='macro', zero_division=0)
        f1 = 2 * (precision * recall) / (precision + recall + 1e-10)
        
        # 多分类 AUC
        try:
            auc = roc_auc_score(
                y_true=targets, 
                y_score=probs, 
                multi_class='ovr',
                average='macro'
            )
        except ValueError:
            auc = 0.0
            logger.warning(f"无法计算{task_name}的AUC，可能是某些类别样本数太少")
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'auc': auc
        }
    
    # 分别计算三个任务的指标
    metrics_1_7 = calculate_task_metrics(all_preds_1_7, all_targets_1_7, all_probs_1_7, "1-7天")
    metrics_8_14 = calculate_task_metrics(all_preds_8_14, all_targets_8_14, all_probs_8_14, "8-14天")
    metrics_15_21 = calculate_task_metrics(all_preds_15_21, all_targets_15_21, all_probs_15_21, "15-21天")
    
    # 计算平均指标
    avg_accuracy = (metrics_1_7['accuracy'] + metrics_8_14['accuracy'] + metrics_15_21['accuracy']) / 3
    avg_precision = (metrics_1_7['precision'] + metrics_8_14['precision'] + metrics_15_21['precision']) / 3
    avg_recall = (metrics_1_7['recall'] + metrics_8_14['recall'] + metrics_15_21['recall']) / 3
    avg_f1 = (metrics_1_7['f1'] + metrics_8_14['f1'] + metrics_15_21['f1']) / 3
    avg_auc = (metrics_1_7['auc'] + metrics_8_14['auc'] + metrics_15_21['auc']) / 3
    
    metrics = {
        'val_loss': avg_val_loss,
        'accuracy': avg_accuracy,
        'precision': avg_precision,
        'recall': avg_recall,
        'f1': avg_f1,
        'auc': avg_auc,
        # 各任务详细指标
        'task_1_7': metrics_1_7,
        'task_8_14': metrics_8_14,
        'task_15_21': metrics_15_21
    }
    
    return avg_val_loss, metrics

if __name__ == "__main__":
    logger.info("开始数据加载和预处理...")
    feature_gen_main = FeatureGenerateMain(
        running_dt=config.TRAIN_RUNNING_DT,
        origin_feature_precompute_interval=config.TRAIN_INTERVAL,
        logger=logger
    )
    feature_gen_main.generate_feature()  # 生成特征

    index_sample_obj = DaysPredictionIndexSampleDataset()
    connect_feature_obj = DaysPredictionFeatureDataset()
    logger.info('----------Create index sample---------')
    train_index_data = index_sample_obj.build_train_dataset(config.TRAIN_RUNNING_DT, config.TRAIN_INTERVAL)
    logger.info("----------Generating train dataset----------")
    train_connect_feature_data = connect_feature_obj.build_train_dataset(input_dataset=train_index_data.copy(), param=None)

    # 生成lable
    label_generator = LabelGenerator(
        feature_data=train_connect_feature_data,
        running_dt=config.TRAIN_RUNNING_DT,
        interval_days=config.TRAIN_INTERVAL
    )
    logger.info("开始生成标签...")
    X, y = label_generator.has_risk_period_generate_multi_label_days_alter()
    logger.info(f"标签计算完成，特征字段为：{X.columns}， 标签数据字段为：{y.columns}")
    logger.info(f"X,y特征数据形状为：{X.shape}， 标签数据形状为：{y.shape}")
    
    # transformer
    if X is None:
        logger.error("特征数据加载失败，程序退出。")
        exit()
        
    train_X, val_X, train_y, val_y = split_data(X, y)
    # 重建索引，不然后面tranform会重建索引导致X与y在concat时不匹配
    train_X.reset_index(drop=True, inplace=True)
    val_X.reset_index(drop=True, inplace=True)
    train_y.reset_index(drop=True, inplace=True)
    val_y.reset_index(drop=True, inplace=True)

    logger.info(f"train_X数据字段为：{train_X.columns}")
    logger.info(f"val_X数据字段为：{val_X.columns}")
    param = {}
    transform = AbortionPredictionTransformPipeline(transform_feature_names = ColumnsConfig.feature_columns) # 传入使用特征

    # 对数据集进行处理
    train_X_transformed = transform.fit_transform(input_dataset=train_X) # 离散数据与连续数据处理
    val_X_transformed = transform.transform(input_dataset=val_X) # 离散数据与连续数据处理
    train_X_transformed.to_csv(
        DataPathConfig.TRAIN_TRANSFORMED_FEATURE_DATA_SAVE_PATH,
        index=False,
        encoding='utf-8-sig'
    )
    val_X_transformed.to_csv(
        DataPathConfig.VAL_TRANSFORMED_FEATURE_DATA_SAVE_PATH,
        index=False,
        encoding='utf-8-sig'
    )

    with open(config.TRANSFORMER_SAVE_PATH, "w+") as dump_file:
        dump_file.write(transform.to_json()) # 保存为json 确保预测时使用与训练时相同的转换参数 比如使用测试集预测时，城市广州在训练时对应的id为2，预测时也为2
    logger.info("transformer transfrom_columns size: %d" % len(transform.features))
    logger.info("Saved transformer to {}".format(config.TRANSFORMER_SAVE_PATH))
    
    if train_X_transformed.isna().any().any(): # 检查是否存在空值
        logger.info("!!!Warning: Nan in train_X")
    if train_y.isna().any().any(): # 检查是否存在空值
        logger.info("!!!Warning: Nan in train_y")


    # 生成mask列
    transformed_masked_null_train_X = mask_feature_null(data=train_X_transformed, mode='train')
    transformed_masked_null_val_X = mask_feature_null(data=val_X_transformed, mode='val')
    transformed_masked_null_train_X.fillna(0, inplace=True)  # 填充空值为0
    transformed_masked_null_val_X.fillna(0, inplace=True)  # 填充空值为0
    logger.info(f"data_transformed_masked_null数据字段为：{transformed_masked_null_train_X.columns}")

    train_df = pd.concat([transformed_masked_null_train_X, train_y], axis=1)
    val_df = pd.concat([transformed_masked_null_val_X, val_y], axis=1)
    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)
    logger.info(f"train_df数据字段为：{train_df.columns}")
    logger.info(f"val_df数据字段为：{val_df.columns}")


    # train_X, train_y = create_sequences(train_data, target_column=ColumnsConfig.HAS_RISK_LABEL, seq_length=config.SEQ_LENGTH, feature_columns=ColumnsConfig.feature_columns)
    # test_X, test_y = create_sequences(val_data, target_column=ColumnsConfig.HAS_RISK_LABEL, seq_length=config.SEQ_LENGTH, feature_columns=ColumnsConfig.feature_columns)
    # logger.info(f"训练集X形状为：{train_X}")
    # logger.info(f"训练集y形状为：{train_y}")
    # logger.info("数据预处理完成.")

    # 5. 创建 PyTorch Dataset 和 DataLoader
    periods = [(1, 7), (8, 14), (15, 21)]
    days_label_list = [ColumnsConfig.DAYS_RISK_8_CLASS_PRE.format(start, end) for start, end in periods]
    train_dataset = DaysDataset(train_df, label=days_label_list)
    val_dataset = DaysDataset(val_df, label=days_label_list)

    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, collate_fn=_collate_fn, num_workers=config.NUM_WORKERS) # Windows下 num_workers>0 可能有问题
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False, collate_fn=_collate_fn, num_workers=config.NUM_WORKERS)
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
    model = Days_NFM(params).to(config.DEVICE) # 等待模型实现
    logger.info("模型初始化完成.")
    logger.info(f"模型结构:\n{model}")

    # criterion = nn.CrossEntropyLoss()  # 假设是回归任务，使用均方误差
    # 根据数据分布设置权重，假设类别0和7频率高
    # class_weights = [0.2, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 0.5]  # 根据实际分布调整
    # criterion = FocalLoss(alpha=class_weights, gamma=2.0)

    task_weights = calculate_all_task_weights(train_y, num_classes=8, method='effective')
    # 创建三个任务的损失函数
    criterion_1_7 = FocalLoss(alpha=task_weights['task_1_7'], gamma=2.0)
    criterion_8_14 = FocalLoss(alpha=task_weights['task_8_14'], gamma=2.0)
    criterion_15_21 = FocalLoss(alpha=task_weights['task_15_21'], gamma=2.0)
    criterions = (criterion_1_7, criterion_8_14, criterion_15_21)

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
    trained_model = train_model(model, train_loader, val_loader, criterions, optimizer, config.NUM_EPOCHS, config.DEVICE, early_stopping=early_stopping)

    # --- 模型评估 (可选，在测试集上) ---
    # ...

    # --- 保存最终模型 (如果未使用早停保存最佳模型) ---
    # final_model_path = os.path.join(config.MODEL_SAVE_PATH, "final_lstm_model.pt")
    # torch.save(trained_model.state_dict(), final_model_path)
    # logger.info(f"最终模型已保存至: {final_model_path}")