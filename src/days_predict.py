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
from dataset.dataset import HasRiskDataset, MultiTaskDataset, DaysDataset
from utils.logger import setup_logger
from utils.early_stopping import EarlyStopping
from utils.save_csv import save_to_csv, read_csv
from feature.gen_feature import FeatureGenerator
from feature.gen_label import LabelGenerator
from transform.transform import FeatureTransformer
from model.mlp import Has_Risk_MLP
from model.nfm import Has_Risk_NFM
from model.multi_task_nfm import Multi_Task_NFM
from model.days_nfm import Days_NFM
from transform.abortion_prediction_transform import AbortionPredictionTransformPipeline
# 设置浮点数显示为小数点后2位，抑制科学计数法
# np.set_printoptions(precision=2, suppress=True)

# 设置随机种子
torch.manual_seed(config.RANDOM_SEED)
np.random.seed(config.RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(config.RANDOM_SEED)

# 初始化日志
logger = setup_logger(logger_config.PREDICT_LOG_FILE_PATH, logger_name="TrainLogger")

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
    features_tensor = torch.tensor(features_list, dtype=torch.float32)
    days_1_7_tensor = torch.tensor(days_label_1_7_list, dtype=torch.long)    # 分类标签用long
    days_8_14_tensor = torch.tensor(days_label_8_14_list, dtype=torch.long)
    days_15_21_tensor = torch.tensor(days_label_15_21_list, dtype=torch.long)
    
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
    elif mode == 'predict':
        save_to_csv(filepath=DataPathConfig.DATA_TRANSFORMED_MASK_NULL_PREDICT_PATH, df=data_transformed_Mask_Null)
    logger.info("特征空值掩码处理完成，数据已保存。")
    return data_transformed_Mask_Null


def predict_model(model, predict_loader, device, predict_index):
    logger.info("开始多任务预测...")
    
    # 加载训练好的模型权重
    if os.path.exists(config.MODEL_SAVE_PATH):
        model.load_state_dict(torch.load(config.MODEL_SAVE_PATH, map_location=device))
        logger.info(f"成功加载模型权重: {config.MODEL_SAVE_PATH}")
    else:
        logger.error(f"未找到模型权重文件: {config.MODEL_SAVE_PATH}")
        return None
    
    # 切换到评估模式
    model.eval()
    
    # 存储三个任务的预测结果
    all_predictions_1_7 = []
    all_probs_1_7 = []
    
    all_predictions_8_14 = []
    all_probs_8_14 = []
    
    all_predictions_15_21 = []
    all_probs_15_21 = []
    
    # 不计算梯度，提高效率
    with torch.no_grad():
        for inputs, _, _, _ in tqdm(predict_loader, desc="预测进度"):
            # 将输入数据移动到指定设备
            inputs = inputs.to(device)
            
            # 前向传播，获取三个任务的模型输出
            days_1_7_output, days_8_14_output, days_15_21_output = model(inputs)
            
            # 处理1-7天任务
            probs_1_7 = torch.softmax(days_1_7_output, dim=1)
            _, predicted_1_7 = torch.max(days_1_7_output, 1)
            all_predictions_1_7.append(predicted_1_7.cpu().numpy())
            all_probs_1_7.append(probs_1_7.cpu().numpy())
            
            # 处理8-14天任务
            probs_8_14 = torch.softmax(days_8_14_output, dim=1)
            _, predicted_8_14 = torch.max(days_8_14_output, 1)
            all_predictions_8_14.append(predicted_8_14.cpu().numpy())
            all_probs_8_14.append(probs_8_14.cpu().numpy())
            
            # 处理15-21天任务
            probs_15_21 = torch.softmax(days_15_21_output, dim=1)
            _, predicted_15_21 = torch.max(days_15_21_output, 1)
            all_predictions_15_21.append(predicted_15_21.cpu().numpy())
            all_probs_15_21.append(probs_15_21.cpu().numpy())
    
    # 合并所有批次的预测结果
    predictions_1_7 = np.concatenate(all_predictions_1_7)
    probabilities_1_7 = np.concatenate(all_probs_1_7)
    
    predictions_8_14 = np.concatenate(all_predictions_8_14)
    probabilities_8_14 = np.concatenate(all_probs_8_14)
    
    predictions_15_21 = np.concatenate(all_predictions_15_21)
    probabilities_15_21 = np.concatenate(all_probs_15_21)

    # 添加安全检查，确保长度一致
    if len(predictions_1_7) != len(predict_index):
        logger.error(f"预测结果长度 ({len(predictions_1_7)}) 与 predict_index 长度 ({len(predict_index)}) 不匹配!")
        return None
    
    logger.info(f"预测完成，共预测 {len(predictions_1_7)} 个样本")
    
    # 将预测结果添加到预测索引DataFrame中
    
    # 1-7天任务预测结果
    predict_index['abort_day_1_7_pred'] = predictions_1_7
    for i in range(probabilities_1_7.shape[1]):
        predict_index[f'prob_abort_day_1_7_pred_{i}'] = probabilities_1_7[:, i]
    
    # 8-14天任务预测结果
    predict_index['abort_day_8_14_pred'] = predictions_8_14
    for i in range(probabilities_8_14.shape[1]):
        predict_index[f'prob_abort_day_1_7_pred_{i}'] = probabilities_8_14[:, i]
    
    # 15-21天任务预测结果
    predict_index['abort_day_15_21_pred'] = predictions_15_21
    for i in range(probabilities_15_21.shape[1]):
        predict_index[f'prob_abort_day_1_7_pred_{i}'] = probabilities_15_21[:, i]
    
    # 输出预测统计信息
    logger.info("=== 预测结果统计 ===")
    logger.info(f"1-7天任务预测类别分布:\n{predict_index['predicted_days_1_7'].value_counts()}")
    logger.info(f"8-14天任务预测类别分布:\n{predict_index['predicted_days_8_14'].value_counts()}")
    logger.info(f"15-21天任务预测类别分布:\n{predict_index['predicted_days_15_21'].value_counts()}")
    
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
    X, y = label_generator.days_period_generate_multi_task_alter()
    X.reset_index(drop=True, inplace=True)
    y.reset_index(drop=True, inplace=True)
    predict_index_label = pd.concat([X, y], axis=1)
    logger.info(f"预测数据的索引为：{predict_index_label.columns}")

    # transform
    if X is None:
        logger.error("特征数据加载失败，程序退出。")
        exit()

    with open(config.TRANSFORMER_SAVE_PATH, "r+") as dump_file:
        transform = AbortionPredictionTransformPipeline.from_json(dump_file.read())
    transformed_feature_df = transform.transform(input_dataset=X)
    # transformer = FeatureTransformer(
    #     discrete_cols=ColumnsConfig.DISCRETE_COLUMNS,
    #     continuous_cols=ColumnsConfig.CONTINUOUS_COLUMNS,
    #     invariant_cols=ColumnsConfig.INVARIANT_COLUMNS,
    #     model_discrete_cols=ColumnsConfig.MODEL_DISCRETE_COLUMNS,
    # )
    # transformer = transformer.load_params(config.TRANSFORMER_SAVE_PATH)
    # transform_dict = transformer.params

    # transformed_feature_df = transformer.transform(X.copy())
    # transformed_feature_df.to_csv(
    #     DataPathConfig.TRANSFORMED_FEATURE_DATA_SAVE_PATH,
    #     index=False,
    #     encoding='utf-8-sig'
    # )
    # logger.info(f"trainsformed_feature_df数据字段为：{transformed_feature_df.columns}")

    # 保存离散特征类别数用于embedding
    # discrete_class_num = transformer.discrete_column_class_count(transformed_feature_df)
    # logger.info(f"离散特征的类别数量: {discrete_class_num}")

    # 预测数据
    predict_X = transformed_feature_df.copy()
    predict_X = predict_X[ColumnsConfig.feature_columns]

    periods = [(1, 7), (8, 14), (15, 21)]
    days_label_list = [ColumnsConfig.DAYS_RISK_8_CLASS_PRE.format(start, end) for start, end in periods]
    predict_y = y.copy()
    predict_y = predict_y[days_label_list]

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

    # predict_index_df = predict_index_label[['stats_dt', 'pigfarm_dk', ColumnsConfig.HAS_RISK_LABEL]].reset_index(drop=True)

    # train_X, train_y = create_sequences(train_data, target_column=ColumnsConfig.HAS_RISK_LABEL, seq_length=config.SEQ_LENGTH, feature_columns=ColumnsConfig.feature_columns)
    # test_X, test_y = create_sequences(val_data, target_column=ColumnsConfig.HAS_RISK_LABEL, seq_length=config.SEQ_LENGTH, feature_columns=ColumnsConfig.feature_columns)
    # logger.info(f"训练集X形状为：{train_X}")
    # logger.info(f"训练集y形状为：{train_y}")
    # logger.info("数据预处理完成.")

    # 5. 创建 PyTorch Dataset 和 DataLoader
    predict_dataset = DaysDataset(predict_df, label=days_label_list)

    predict_loader = DataLoader(predict_dataset, batch_size=config.BATCH_SIZE, shuffle=False, collate_fn=_collate_fn,num_workers=config.NUM_WORKERS) # Windows下 num_workers>0 可能有问题
    logger.info("数据加载器准备完毕.")

    # --- 模型、损失函数、优化器 ---
    feature_dict = transform.features.features
    Categorical_feature = ColumnsConfig.DISCRETE_COLUMNS # 离散值字段
    logger.info(f"pigfarm_dk类别数：{feature_dict[Categorical_feature[0]].category_encode.size}")
    params = {
        'model_discrete_columns': ColumnsConfig.MODEL_DISCRETE_COLUMNS,
        'model_continuous_columns': ColumnsConfig.MODEL_CONTINUOUS_COLUMNS,
        'dropout': config.DROPOUT,

        'pigfarm_dk': feature_dict[Categorical_feature[0]].category_encode.size,
        'city': feature_dict[Categorical_feature[1]].category_encode.size,
        'month': 12,
        'season': 4,
    }
    model = Days_NFM(params).to(config.DEVICE) # 等待模型实现
    logger.info("模型初始化完成.")
    logger.info(f"模型结构:\n{model}")

    # --- 开始训练 (当前被注释掉，因为模型未定义) ---
    predict_df = predict_model(model, predict_loader, config.DEVICE, predict_index_label)

    # 保存预测结果
    save_to_csv(df=predict_df, filepath=config.main_predict.DAYS_PREDICT_RESULT_SAVE_PATH)
    logger.info(f"预测结果已保存至: {config.main_predict.HAS_RISK_PREDICT_RESULT_SAVE_PATH}")
