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
from dataset.dataset import HasRiskDataset
from utils.logger import setup_logger
from utils.early_stopping import EarlyStopping
from utils.save_csv import save_to_csv, read_csv
from feature.gen_feature import FeatureGenerator
from feature.gen_label import LabelGenerator
from transform.transform import FeatureTransformer
from model.mlp import Has_Risk_MLP
from model.nfm import Has_Risk_NFM

# 设置浮点数显示为小数点后2位，抑制科学计数法
# np.set_printoptions(precision=2, suppress=True)

# 设置随机种子
torch.manual_seed(config.RANDOM_SEED)
np.random.seed(config.RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(config.RANDOM_SEED)

# 初始化日志
logger = setup_logger(logger_config.PREDICT_LOG_FILE_PATH, logger_name="TrainLogger")

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
    elif mode == 'predict':
        save_to_csv(filepath=DataPathConfig.DATA_TRANSFORMED_MASK_NULL_PREDICT_PATH, df=data_transformed_Mask_Null)
    logger.info("特征空值掩码处理完成，数据已保存。")
    return data_transformed_Mask_Null


def predict_model(model, predict_loader, device, predict_index):
    logger.info("开始预测...")
    
    # 加载训练好的模型权重
    if os.path.exists(config.MODEL_SAVE_PATH):
        model.load_state_dict(torch.load(config.MODEL_SAVE_PATH, map_location=device))
        logger.info(f"成功加载模型权重: {config.MODEL_SAVE_PATH}")
    else:
        logger.error(f"未找到模型权重文件: {config.MODEL_SAVE_PATH}")
        return None
    
    # 切换到评估模式
    model.eval()
    
    # 存储预测结果
    all_predictions = []
    all_probs = []
    
    # 不计算梯度，提高效率
    with torch.no_grad():
        for inputs, _ in tqdm(predict_loader, desc="预测进度"):
            # 将输入数据移动到指定设备
            inputs = inputs.to(device)
            
            # 前向传播，获取模型输出
            outputs = model(inputs)
            
            # 应用softmax获取概率
            probs = torch.softmax(outputs, dim=1)
            
            # 获取预测类别
            _, predicted = torch.max(outputs, 1)
            
            # 将张量转移到CPU并转换为NumPy数组
            probs_np = probs.cpu().numpy()
            predicted_np = predicted.cpu().numpy()
            
            # 添加到结果列表
            all_predictions.append(predicted_np)
            all_probs.append(probs_np)
    
    # 合并所有批次的预测结果
    predictions = np.concatenate(all_predictions)
    probabilities = np.concatenate(all_probs)

    # 添加安全检查，确保长度一致
    if len(predictions) != len(predict_index):
        logger.error(f"预测结果长度 ({len(predictions)}) 与 predict_index 长度 ({len(predict_index)}) 不匹配!")
        return None
    
    logger.info(f"预测完成，共预测 {len(predictions)} 个样本")
    
    # 将预测结果添加到预测索引DataFrame中
    predict_index['predicted_class'] = predictions
    
    # 添加四个类别的概率列
    for i in range(4):
        predict_index[f'prob_class_{i}'] = probabilities[:, i]
    
    # 输出预测统计信息
    logger.info(f"预测类别分布:\n{predict_index['predicted_class'].value_counts()}")
    
    return predict_index

if __name__ == "__main__":
    logger.info("开始数据加载和预处理...")
    # 1. 加载和基础预处理数据
    # 生成特征
    feature_generator = FeatureGenerator(running_dt=config.main_predict.PREDICT_RUNNING_DT, interval_days=config.main_predict.PREDICT_INTERVAL)
    feature_df = feature_generator.generate_features()

     # 生成lable
    label_generator = LabelGenerator(
        feature_data=feature_df,
        running_dt=config.main_predict.PREDICT_RUNNING_DT,
        interval_days=config.main_predict.PREDICT_INTERVAL,
    )
    logger.info("开始生成标签...")
    X, y = label_generator.has_risk_4_class_generate_label()
    predict_index_label = pd.concat([X, y], axis=1)
    logger.info(f"预测数据的索引为：{predict_index_label.columns}")

    # transform
    if X is None:
        logger.error("特征数据加载失败，程序退出。")
        exit()
    transformer = FeatureTransformer(
        discrete_cols=ColumnsConfig.DISCRETE_COLUMNS,
        continuous_cols=ColumnsConfig.CONTINUOUS_COLUMNS,
        invariant_cols=ColumnsConfig.INVARIANT_COLUMNS,
        model_discrete_cols=ColumnsConfig.MODEL_DISCRETE_COLUMNS,
    )
    transformer = transformer.load_params(config.TRANSFORMER_SAVE_PATH)
    transform_dict = transformer.params

    transformed_feature_df = transformer.transform(X.copy())
    transformed_feature_df.to_csv(
        DataPathConfig.TRANSFORMED_FEATURE_DATA_SAVE_PATH,
        index=False,
        encoding='utf-8-sig'
    )
    logger.info(f"trainsformed_feature_df数据字段为：{transformed_feature_df.columns}")
    # 保存离散特征类别数用于embedding
    # discrete_class_num = transformer.discrete_column_class_count(transformed_feature_df)
    # logger.info(f"离散特征的类别数量: {discrete_class_num}")

    # 预测数据
    predict_X = transformed_feature_df.copy()
    predict_y = y.copy()
    predict_X = predict_X[ColumnsConfig.feature_columns]
    predict_y = predict_y[ColumnsConfig.HAS_RISK_LABEL]

    # 生成mask列
    transformed_masked_null_predict_X = mask_feature_null(data=predict_X, mode='predict')

    transformed_masked_null_predict_X.fillna(0, inplace=True)  # 填充空值为0
    logger.info(f"data_transformed_masked_null数据字段为：{transformed_masked_null_predict_X.columns}")
    predict_df = pd.concat([transformed_masked_null_predict_X, predict_y], axis=1)
    predict_df = predict_df.reset_index(drop=True)
    logger.info(f"预测数据的索引为：{predict_df.columns}")
    predict_df.to_csv(
        DataPathConfig.PREDICT_DATA_TRANSFORMED_MASK_NULL_PREDICT_PATH,
        index=False,
        encoding='utf-8-sig'
    )
    predict_index_df = transformed_feature_df[['stats_dt', 'pigfarm_dk']].reset_index(drop=True)
    # predict_index_df = predict_index_label[['stats_dt', 'pigfarm_dk', ColumnsConfig.HAS_RISK_LABEL]].reset_index(drop=True)

    # train_X, train_y = create_sequences(train_data, target_column=ColumnsConfig.HAS_RISK_LABEL, seq_length=config.SEQ_LENGTH, feature_columns=ColumnsConfig.feature_columns)
    # test_X, test_y = create_sequences(val_data, target_column=ColumnsConfig.HAS_RISK_LABEL, seq_length=config.SEQ_LENGTH, feature_columns=ColumnsConfig.feature_columns)
    # logger.info(f"训练集X形状为：{train_X}")
    # logger.info(f"训练集y形状为：{train_y}")
    # logger.info("数据预处理完成.")

    # 5. 创建 PyTorch Dataset 和 DataLoader
    predict_dataset = HasRiskDataset(predict_df, label=ColumnsConfig.HAS_RISK_LABEL)

    predict_loader = DataLoader(predict_dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=0) # Windows下 num_workers>0 可能有问题
    logger.info("数据加载器准备完毕.")

    # --- 模型、损失函数、优化器 ---
    params = {
        'model_discrete_columns': ColumnsConfig.MODEL_DISCRETE_COLUMNS,
        'model_continuous_columns': ColumnsConfig.MODEL_CONTINUOUS_COLUMNS,
        'dropout': config.DROPOUT,

        'pigfarm_dk': len(transform_dict['discrete_mappings']['pigfarm_dk']['key2id']),
        'month': 12,
        'is_single': 2,
    }
    model = Has_Risk_MLP(params).to(config.DEVICE) # 等待模型实现
    logger.info("模型初始化完成.")
    logger.info(f"模型结构:\n{model}")

    # --- 开始训练 (当前被注释掉，因为模型未定义) ---
    predict_df = predict_model(model, predict_loader, config.DEVICE, predict_index_label)

    # 保存预测结果
    save_to_csv(df=predict_df, filepath=config.main_predict.HAS_RISK_PREDICT_RESULT_SAVE_PATH)
    logger.info(f"预测结果已保存至: {config.main_predict.HAS_RISK_PREDICT_RESULT_SAVE_PATH}")
