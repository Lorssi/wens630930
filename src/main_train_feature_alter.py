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
def sample_probability(labels):
    """
    为不平衡的数据集创建加权随机采样器
    
    Args:
        labels (np.ndarray): 标签数组，已经是numpy格式
        
    Returns:
        WeightedRandomSampler: 基于类别权重的采样器
    """
    # 确保输入是numpy数组
    if not isinstance(labels, np.ndarray):
        labels = np.array(labels)
    labels = labels.astype(int)  # 确保标签是整数类型

    # 统计每个类别的样本数量
    unique_classes = np.unique(labels)
    class_sample_count = np.array([len(np.where(labels == t)[0]) for t in unique_classes])
    
    # 计算类别权重（样本量少的类别权重大）
    weight = 1. / class_sample_count
    
    # 为每个样本分配权重
    samples_weight = np.array([weight[np.where(unique_classes == t)[0][0]] for t in labels])
    
    # 转换为PyTorch张量
    samples_weight = torch.from_numpy(samples_weight)
    samples_weight = samples_weight.double()
    
    # 创建采样器
    sampler = WeightedRandomSampler(
        weights=samples_weight,
        num_samples=len(samples_weight),
        replacement=True
    )
    
    return sampler

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device, early_stopping=None):
    logger.info("开始训练...")
    # scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.8, patience=1)
    
    best_val_metrics = None
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for batch_idx, (features, targets) in enumerate(train_loader):
            features, targets = features.to(device), targets.to(device)  # 将特征和标签移至目标设备
            # label

            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, targets)
            loss.backward()

            optimizer.step()

            train_loss += loss.item()

            # if batch_idx % 100 == 0: # 每100个batch打印一次日志
            #     logger.info(f"Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx+1}/{len(train_loader)}], Loss: {loss.item():.4f}")

        avg_train_loss = train_loss / len(train_loader)

        # --- 验证阶段 ---
        avg_val_loss, metrics = eval(model, val_loader, criterion, device)
        
        # 打印训练指标
        logger.info(f"Epoch [{epoch+1}/{num_epochs}], 训练集平均损失：{avg_train_loss:.4f}, 验证集平均损失: {avg_val_loss:.4f}")
        logger.info(f"Epoch [{epoch+1}/{num_epochs}], 准确率: {metrics['accuracy']:.4f}, 精确率: {metrics['precision']:.4f}, 召回率: {metrics['recall']:.4f}, F1: {metrics['f1']:.4f}, AUC: {metrics['auc']:.4f}")
        
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
    avg_train_loss, train_metrics = eval(model, train_loader, criterion, device)
    avg_val_loss, val_metrics = eval(model, val_loader, criterion, device)
    
    # --- 打印最终评估结果 ---
    logger.info(f"最终评估 - 训练集损失: {avg_train_loss:.4f}, 验证集损失: {avg_val_loss:.4f}")
    logger.info(f"最终评估 - 训练集: 准确率: {train_metrics['accuracy']:.4f}, 精确率: {train_metrics['precision']:.4f}, 召回率: {train_metrics['recall']:.4f}, F1: {train_metrics['f1']:.4f}, AUC: {train_metrics['auc']:.4f}")
    logger.info(f"最终评估 - 验证集: 准确率: {val_metrics['accuracy']:.4f}, 精确率: {val_metrics['precision']:.4f}, 召回率: {val_metrics['recall']:.4f}, F1: {val_metrics['f1']:.4f}, AUC: {val_metrics['auc']:.4f}")
    
    logger.info("训练完成.")
    return model

# def eval(model, val_loader, criterion, device):
#     """
#     评估模型在验证集上的性能（支持连续值标签和二分类标签混合）
#     """
#     model.eval()
#     val_loss = 0.0
#     all_preds = []
#     all_targets = []
#     all_probs = []
    
#     with torch.no_grad():
#         for features, targets in val_loader:
#             # 获取批次数据并移至目标设备
#             features, targets = features.to(device), targets.to(device)
            
#             # 前向传播
#             outputs = model(features)
#             loss = criterion(outputs, targets)
#             val_loss += loss.item()
            
#             # 获取预测结果 - 使用sigmoid
#             probs = torch.sigmoid(outputs)
#             preds = (probs > 0.5).float()  # 二值化，阈值为0.5
            
#             # 收集结果
#             all_preds.append(preds.cpu().numpy())
#             all_targets.append(targets.cpu().numpy())
#             all_probs.append(probs.cpu().numpy())
    
#     # 计算平均损失
#     avg_val_loss = val_loss / len(val_loader)
    
#     # 转换为numpy数组
#     all_preds = np.vstack(all_preds)
#     all_targets = np.vstack(all_targets)
#     all_probs = np.vstack(all_probs)
    
#     # 为评估指标，将真实标签二值化（>0.5为1，否则为0）
#     binary_targets = (all_targets > 0.5).astype(int)
    
#     # 计算多标签指标（使用二值化后的标签）
#     accuracy = accuracy_score(binary_targets, all_preds)
#     precision = precision_score(binary_targets, all_preds, average='macro', zero_division=0)
#     recall = recall_score(binary_targets, all_preds, average='macro', zero_division=0)
#     f1 = 2 * (precision * recall) / (precision + recall + 1e-10)
    
#     # 多标签 AUC
#     try:
#         # 对每个标签分别计算ROC AUC，然后取平均
#         auc_scores = []
#         for i in range(all_probs.shape[1]):
#             # 只有当某个标签既有正例又有负例时才计算AUC
#             if len(np.unique(binary_targets[:, i])) > 1:
#                 auc_scores.append(roc_auc_score(binary_targets[:, i], all_probs[:, i]))
        
#         auc = np.mean(auc_scores) if auc_scores else 0.0
#     except ValueError:
#         auc = 0.0
#         logger.warning("无法计算AUC，可能是某些类别样本数太少")
    
#     # 额外添加连续值评估指标（均方误差）
#     mse = np.mean((all_probs - all_targets) ** 2)
    
#     metrics = {
#         'val_loss': avg_val_loss,
#         'accuracy': accuracy,
#         'precision': precision,
#         'recall': recall,
#         'f1': f1,
#         'auc': auc,
#         'mse': mse  # 添加均方误差指标
#     }
    
#     return avg_val_loss, metrics

def eval(model, val_loader, criterion, device):
    """
    评估模型在验证集上的性能（多标签版本）
    """
    model.eval()
    val_loss = 0.0
    all_preds = []
    all_targets = []
    all_probs = []
    
    with torch.no_grad():
        for features, targets in val_loader:
            # 获取批次数据并移至目标设备
            features, targets = features.to(device), targets.to(device)  # 将特征和标签移至目标设备
            
            # 前向传播
            outputs = model(features)
            loss = criterion(outputs, targets)
            val_loss += loss.item()
            
            # 获取预测结果 - 使用sigmoid而非softmax
            probs = torch.sigmoid(outputs)
            preds = (probs > 0.5).float()  # 二值化，阈值为0.5
            
            # 收集结果
            all_preds.append(preds.cpu().numpy())
            all_targets.append(targets.cpu().numpy())
            all_probs.append(probs.cpu().numpy())
    
    # 计算平均损失
    avg_val_loss = val_loss / len(val_loader)
    
    # 转换为numpy数组
    all_preds = np.vstack(all_preds)
    all_targets = np.vstack(all_targets)
    all_probs = np.vstack(all_probs)
    
    # 计算多标签指标
    accuracy = accuracy_score(all_targets, all_preds)
    precision = precision_score(all_targets, all_preds, average='macro', zero_division=0)
    recall = recall_score(all_targets, all_preds, average='macro', zero_division=0)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-10)
    
    # 多标签 AUC
    try:
        # 对每个标签分别计算ROC AUC，然后取平均
        auc_scores = []
        for i in range(all_probs.shape[1]):
            # 只有当某个标签既有正例又有负例时才计算AUC
            if len(np.unique(all_targets[:, i])) > 1:
                auc_scores.append(roc_auc_score(all_targets[:, i], all_probs[:, i]))
        
        auc = np.mean(auc_scores) if auc_scores else 0.0
    except ValueError:
        auc = 0.0
        logger.warning("无法计算AUC，可能是某些类别样本数太少")
    
    metrics = {
        'val_loss': avg_val_loss,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc': auc
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
    transform_param = {}
    transform = AbortionPredictionTransformPipeline(transform_feature_names = ColumnsConfig.feature_columns, feature_groups=transform_param) # 传入使用特征

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
    train_df.to_csv(
        "train_df.csv",
        index=False,
        encoding='utf-8-sig'
    )


    # train_X, train_y = create_sequences(train_data, target_column=ColumnsConfig.HAS_RISK_LABEL, seq_length=config.SEQ_LENGTH, feature_columns=ColumnsConfig.feature_columns)
    # test_X, test_y = create_sequences(val_data, target_column=ColumnsConfig.HAS_RISK_LABEL, seq_length=config.SEQ_LENGTH, feature_columns=ColumnsConfig.feature_columns)
    # logger.info(f"训练集X形状为：{train_X}")
    # logger.info(f"训练集y形状为：{train_y}")
    # logger.info("数据预处理完成.")

    # 5. 创建 PyTorch Dataset 和 DataLoader
    periods = [(1, 7), (8, 14), (15, 21)]
    has_risk_label_list = [ColumnsConfig.HAS_RISK_4_CLASS_PRE.format(left, right) for left, right in periods]
    train_dataset = MultiTaskAndMultiLabelDataset(train_df, label=has_risk_label_list)
    val_dataset = MultiTaskAndMultiLabelDataset(val_df, label=has_risk_label_list)

    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, collate_fn=_collate_fn,num_workers=config.NUM_WORKERS) # Windows下 num_workers>0 可能有问题
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False, collate_fn=_collate_fn,num_workers=config.NUM_WORKERS)
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
    trained_model = train_model(model, train_loader, val_loader, criterion, optimizer, config.NUM_EPOCHS, config.DEVICE, early_stopping=early_stopping)


    # --- 模型评估 (可选，在测试集上) ---
    # ...

    # --- 保存最终模型 (如果未使用早停保存最佳模型) ---
    # final_model_path = os.path.join(config.MODEL_SAVE_PATH, "final_lstm_model.pt")
    # torch.save(trained_model.state_dict(), final_model_path)
    # logger.info(f"最终模型已保存至: {final_model_path}")

