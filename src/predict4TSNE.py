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
from configs.pigfarm_risk_prediction_config import TrainModuleConfig, PredictModuleConfig, EvalModuleConfig, algo_interim_dir
import config
from dataset.dataset import HasRiskDataset, MultiTaskAndMultiLabelDataset, MultiTaskAndMultiLabelPredictDataset
from utils.logger import setup_logger
from utils.early_stopping import EarlyStopping
from utils.save_csv import save_to_csv, read_csv
from feature.gen_feature import FeatureGenerator
from feature.gen_label import LabelGenerator
from transform.transform import FeatureTransformer
from model.mlp import Has_Risk_MLP
from model.nfm import Has_Risk_NFM, Has_Risk_NFM_MultiLabel, Has_Risk_NFM_MultiLabel_7d1Linear
from model.risk_wider_nfm import Has_Risk_NFM_MultiLabel_Wider
from transform.abortion_prediction_transform import AbortionPredictionTransformPipeline

from dataset.risk_prediction_index_sample_dataset import RiskPredictionIndexSampleDataset
from dataset.risk_prediction_feature_dataset import RiskPredictionFeatureDataset
from module.future_generate_main import FeatureGenerateMain

import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
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
    elif mode == 'predict':
        save_to_csv(filepath=DataPathConfig.DATA_TRANSFORMED_MASK_NULL_PREDICT_PATH, df=data_transformed_Mask_Null)
    logger.info("特征空值掩码处理完成，数据已保存。")
    return data_transformed_Mask_Null

def visualize_tsne(features, labels=None, perplexity=30, n_components=2, title='t-SNE 特征可视化', label_info=''):
    """
    使用t-SNE对特征进行降维并可视化
    
    参数:
    features: 高维特征向量
    labels: 对应的标签，如果有的话
    perplexity: t-SNE的困惑度参数
    n_components: 降维后的维度
    title: 图表标题
    label_info: 标签信息，用于文件命名
    """
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

    logger.info(f"开始进行t-SNE降维，原始特征维度: {features.shape}")
    
    # 执行t-SNE降维
    tsne = TSNE(n_components=n_components, 
                perplexity=perplexity, 
                random_state=config.RANDOM_SEED,
                max_iter=1000,
                learning_rate="auto",
                init="pca")
    
    # 降维计算
    tsne_results = tsne.fit_transform(features)
    logger.info(f"t-SNE降维完成，降维后形状: {tsne_results.shape}")
    
    # 创建可视化
    plt.figure(figsize=(10, 8))
    
    if labels is not None and len(labels) > 0:
        # 如果有标签数据，使用标签颜色
        scatter = plt.scatter(tsne_results[:, 0], tsne_results[:, 1], 
                    c=labels, cmap='viridis', alpha=0.8, s=30)
        plt.colorbar(scatter, label='类别')
    # else:
    #     # 如果没有标签，使用密度图
    #     sns.kdeplot(x=tsne_results[:, 0], y=tsne_results[:, 1], 
    #                cmap="viridis", fill=True, thresh=0.05)
    #     plt.scatter(tsne_results[:, 0], tsne_results[:, 1], 
    #                alpha=0.3, s=10, color='black')
    
    # 使用传入的标题而非硬编码标题
    plt.title(title, fontsize=16)
    plt.xlabel('t-SNE 维度 1', fontsize=14)
    plt.ylabel('t-SNE 维度 2', fontsize=14)
    plt.grid(alpha=0.3)
    
    # 确保输出目录存在
    os.makedirs(DataPathConfig.TSNE_DATA_DIR, exist_ok=True)
    
    # 保存结果
    plt.tight_layout()
    
    # 保存可视化结果，文件名中包含标签信息
    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    # 在文件名中添加标签信息
    file_name = f"tsne_visualization_{label_info}_{timestamp}_{12}.png"
    output_path = os.path.join(DataPathConfig.TSNE_DATA_DIR, file_name)
    plt.savefig(output_path, dpi=300)
    logger.info(f"t-SNE可视化已保存至: {output_path}")
    
    # 同时保存降维后的数据
    tsne_data = pd.DataFrame(
        tsne_results, 
        columns=[f'tsne_dim_{i+1}' for i in range(n_components)]
    )
    
    # 如果有标签，也保存标签
    if labels is not None and len(labels) > 0:
        tsne_data['label'] = labels
    
    # 在CSV文件名中也包含标签信息
    data_file_name = f"tsne_data_{label_info}_{timestamp}.csv"
    data_path = os.path.join(DataPathConfig.TSNE_DATA_DIR, data_file_name)
    tsne_data.to_csv(data_path, index=False)
    logger.info(f"t-SNE数据已保存至: {data_path}")
    
    return tsne_results


def predict_model(model, predict_loader, device):
    logger.info("开始多标签预测...")
    
    # 加载训练好的模型权重
    if os.path.exists(config.MODEL_SAVE_PATH):
        model.load_state_dict(torch.load(config.MODEL_SAVE_PATH, map_location=device))
        logger.info(f"成功加载模型权重: {config.MODEL_SAVE_PATH}")
    else:
        logger.error(f"未找到模型权重文件: {config.MODEL_SAVE_PATH}")
        return None

    features_list = []
    labels_list = []
    
    # 保存特征的变量
    hidden_features = []

    # 定义钩子函数
    def hook_fn(module, input, output):
        hidden_features.append(output.detach().cpu())
    
    # 注册钩子到模型的最后一个隐藏层
    # 获取MLP中倒数第二层（即输出层之前的那一层）
    # 假设你的模型为model，要获取mlp中输出层之前的8维向量
    hook_handle = model.mlp[-3].register_forward_hook(hook_fn)  # -3是获取最后一个激活函数之前的线性层输出

    # 切换到评估模式
    model.eval()
    # 不计算梯度，提高效率
    with torch.no_grad():
        for inputs, labels in tqdm(predict_loader, desc="预测进度"):
            # 将输入数据移动到指定设备
            inputs = inputs.to(device)
            
            # 前向传播，获取模型输出
            _ = model(inputs)
            
            # 收集这个批次的特征
            features_list.append(hidden_features[-1])
            labels_list.append(labels.cpu())
            
            # 清空，准备下一批次
            hidden_features.clear()

    # 移除钩子
    hook_handle.remove()
    
    # 合并所有批次的特征
    all_features = torch.cat(features_list, dim=0).numpy()
    all_labels = torch.cat(labels_list, dim=0).numpy()
    
    return all_features, all_labels

if __name__ == "__main__":
    logger.info("开始数据加载和预处理...")
    index_df = pd.read_csv(config.main_predict.PREDICT_INDEX_TABLE, encoding='utf-8-sig')
    if index_df is None:
        logger.error("索引数据加载失败，程序退出。")
        exit()
    logger.info(f"索引数据从{config.main_predict.PREDICT_INDEX_TABLE}加载成功，数据行数：{len(index_df)}")
    logger.info(f"索引数据的列为：{index_df.columns}")
    index_df['stats_dt'] = pd.to_datetime(index_df['stats_dt'], format='%Y-%m-%d', errors='coerce')

    max_date = index_df['stats_dt'].max()
    min_date = index_df['stats_dt'].min()
    logger.info(f"索引数据的最小日期为：{min_date}, 最大日期为：{max_date}")
    predict_running_dt = max_date + pd.Timedelta(days=22)  # 预测运行日期为最大日期的下一天
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

    connect_feature_obj = RiskPredictionFeatureDataset()
    logger.info("----------Generating train dataset----------")
    prdict_connect_feature_data = connect_feature_obj.build_train_dataset(input_dataset=index_df.copy(), param=None)
    logger.info(f"预测特征数据的行数为：{len(prdict_connect_feature_data)}")

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

    # 生成lable
    label_generator = LabelGenerator(
        feature_data=predict_connect_feature_data,
        running_dt=predict_running_dt.strftime('%Y-%m-%d'),
        interval_days=predict_interval - 1
    )
    logger.info("开始生成标签...")
    predict_connect_feature_data, y = label_generator.has_risk_period_generate_multi_label_alter_nodays()
    predict_connect_feature_data = predict_connect_feature_data.reset_index(drop=True)
    y = y.reset_index(drop=True)
    logger.info(f"标签生成完成，标签数据的行数为：{len(y)}")

    index_merge_df = predict_connect_feature_data[ColumnsConfig.feature_columns]
    index_merge_df.to_csv('predict_index_merge_df.csv', index=False, encoding='utf-8-sig')
    with open(config.TRANSFORMER_SAVE_PATH, "r+") as dump_file:
        transform = AbortionPredictionTransformPipeline.from_json(dump_file.read())
    transformed_feature_df = transform.transform(input_dataset=index_merge_df)
    logger.info(f"特征转换完成，转换后的数据行数为：{len(transformed_feature_df)}")

    predict_X = transformed_feature_df.copy()
    predict_X = predict_X[ColumnsConfig.feature_columns]

    # 生成mask列
    transformed_masked_null_predict_X = mask_feature_null(data=predict_X, mode='predict')
    transformed_masked_null_predict_X.fillna(0, inplace=True)  # 填充空值为0
    logger.info(f"data_transformed_masked_null数据字段为：{transformed_masked_null_predict_X.columns}")
    logger.info(f"data_transformed_masked_null数据行数为：{len(transformed_masked_null_predict_X)}")
    
    transformed_masked_null_predict_X.to_csv(
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
    periods = [(1, 7), (8, 14), (15, 21)]
    has_risk_label_list = [ColumnsConfig.HAS_RISK_4_CLASS_PRE.format(left, right) for left, right in periods]
    df = pd.concat([transformed_masked_null_predict_X, y], axis=1)
    predict_dataset = MultiTaskAndMultiLabelDataset(df=df, label=has_risk_label_list)

    predict_loader = DataLoader(predict_dataset, batch_size=config.BATCH_SIZE, shuffle=False, collate_fn=_collate_fn,num_workers=config.NUM_WORKERS) # Windows下 num_workers>0 可能有问题
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

    # --- 开始训练 (当前被注释掉，因为模型未定义) ---
    all_features, all_labels = predict_model(model, predict_loader, config.DEVICE)

    # 对提取的特征进行t-SNE分析
    # tsne_results = visualize_tsne(features=all_features, labels=all_labels, perplexity=30)
    # 为每个标签创建单独的可视化
    for i, period in enumerate(periods):
        # 提取当前时期的标签
        current_labels = all_labels[:, i]
        
        # 创建此时期的可视化
        period_name = f"{period[0]}-{period[1]}天"
        
        # 为文件名定义标签信息
        label_info = f"period_{period[0]}to{period[1]}"
        
        tsne_results = visualize_tsne(
            features=all_features, 
            labels=current_labels,
            perplexity=40,
            title=f"t-SNE 特征可视化 ({period_name})",
            label_info=label_info  # 添加标签信息用于文件命名
        )