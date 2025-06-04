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
from dataset.dataset import HasRiskDataset, MultiTaskDataset, DaysPredictDataset
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
logger = setup_logger(logger_config.PREDICT_LOG_FILE_PATH, logger_name="TrainLogger")

def _collate_fn(batch):
    # 初始化列表
    features_list = []
    
    for feature in batch:
        # 添加特征
        features_list.append(feature)
    
    # 转换为张量
    features_tensor = torch.tensor(np.array(features_list), dtype=torch.float32)
    
    # 返回特征和所有标签
    return features_tensor

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
        for inputs in tqdm(predict_loader, desc="预测进度"):
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
    predict_index['abort_days_1_7'] = predictions_1_7
    for i in range(probabilities_1_7.shape[1]):
        predict_index[f'prob_abort_day_1_7_pred_{i}'] = probabilities_1_7[:, i]
    
    # 8-14天任务预测结果
    predict_index['abort_days_8_14'] = predictions_8_14
    for i in range(probabilities_8_14.shape[1]):
        predict_index[f'prob_abort_day_8_14_pred_{i}'] = probabilities_8_14[:, i]
    
    # 15-21天任务预测结果
    predict_index['abort_days_15_21'] = predictions_15_21
    for i in range(probabilities_15_21.shape[1]):
        predict_index[f'prob_abort_day_15_21_pred_{i}'] = probabilities_15_21[:, i]
    
    # 输出预测统计信息
    logger.info("=== 预测结果统计 ===")
    logger.info(f"1-7天任务预测类别分布:\n{predict_index['abort_days_1_7'].value_counts()}")
    logger.info(f"8-14天任务预测类别分布:\n{predict_index['abort_days_8_14'].value_counts()}")
    logger.info(f"15-21天任务预测类别分布:\n{predict_index['abort_days_15_21'].value_counts()}")
    
    return predict_index

if __name__ == "__main__":
    logger.info("开始数据加载和预处理...")
    index_df = pd.read_csv(config.main_predict.PREDICT_INDEX_TABLE, encoding='utf-8-sig')
    if index_df is None:
        logger.error("索引数据加载失败，程序退出。")
        exit()
    logger.info(f"索引数据的列为：{index_df.columns}")
    index_df['stats_dt'] = pd.to_datetime(index_df['stats_dt'], format='%Y-%m-%d', errors='coerce')

    max_date = index_df['stats_dt'].max()
    min_date = index_df['stats_dt'].min()
    logger.info(f"索引数据的最小日期为：{min_date}, 最大日期为：{max_date}")
    predict_running_dt = max_date + pd.Timedelta(days=1)  # 预测运行日期为最大日期的下一天
    predict_interval = index_df['stats_dt'].nunique()  # 预测区间为索引数据的唯一日期数
    logger.info(f"预测运行日期为：{predict_running_dt}, 预测区间为：{predict_interval}天")

    # 1. 加载和基础预处理数据
    # 生成特征
    feature_gen_main = FeatureGenerateMain(
        running_dt=predict_running_dt.strftime('%Y-%m-%d'),
        origin_feature_precompute_interval=predict_interval - 1,
        logger=logger
    )
    feature_gen_main.generate_feature()  # 生成特征


    connect_feature_obj = DaysPredictionFeatureDataset()
    logger.info("----------Generating train dataset----------")
    prdict_connect_feature_data = connect_feature_obj.build_train_dataset(input_dataset=index_df.copy(), param=None)

    predict_connect_feature_data = prdict_connect_feature_data.reset_index(drop=True)
    logger.info(f"预测特征数据的列为：{predict_connect_feature_data.columns}")

    predict_index_label = index_df.copy()
    predict_connect_feature_data.to_csv(
        DataPathConfig.PREDICT_INDEX_MERGE_FEATURE_DATA_SAVE_PATH,
        index=False,
        encoding='utf-8-sig'
    )


    # transform
    if predict_connect_feature_data is None:
        logger.error("特征数据加载失败，程序退出。")
        exit()

    index_merge_df = predict_connect_feature_data[ColumnsConfig.feature_columns]
    with open(config.TRANSFORMER_SAVE_PATH, "r+") as dump_file:
        transform = AbortionPredictionTransformPipeline.from_json(dump_file.read())
    transformed_feature_df = transform.transform(input_dataset=index_merge_df)

    predict_X = transformed_feature_df.copy()
    predict_X = predict_X[ColumnsConfig.feature_columns]

    # 生成mask列
    transformed_masked_null_predict_X = mask_feature_null(data=predict_X, mode='predict')
    transformed_masked_null_predict_X.fillna(0, inplace=True)  # 填充空值为0
    logger.info(f"data_transformed_masked_null数据字段为：{transformed_masked_null_predict_X.columns}")
    
    transformed_masked_null_predict_X.to_csv(
        DataPathConfig.PREDICT_DATA_TRANSFORMED_MASK_NULL_PREDICT_PATH,
        index=False,
        encoding='utf-8-sig'
    )

    # 5. 创建 PyTorch Dataset 和 DataLoader
    predict_dataset = DaysPredictDataset(transformed_masked_null_predict_X)

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
