# main_train.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd # 导入 pandas 用于 DataFrame 操作
import os
import json # 导入 json 用于保存/加载参数
from pathlib import Path # 导入 Path 用于路径操作

# 从项目中导入模块
from configs.logger_config import logger_config # 假设此文件存在
from dataset.dataset import HasRiskDataset # 假设此文件存在
from utils.logger import setup_logger # 假设此文件存在
from utils.early_stopping import EarlyStopping # 假设此文件存在
from feature.gen_feature import FeatureGenerator # 假设此文件存在
# from models.lstm_model import YourLSTMModel # 等待模型定义

# 设置随机种子 (假设 config 对象存在)
# class DummyConfig: # Config 的占位符
# RANDOM_SEED = 42
#     # ... 其他配置
# config = DummyConfig()
# torch.manual_seed(config.RANDOM_SEED)
# np.random.seed(config.RANDOM_SEED)
# if torch.cuda.is_available():
#     torch.cuda.manual_seed_all(config.RANDOM_SEED)

# 初始化日志
logger = setup_logger(logger_config.TRAIN_LOG_FILE_PATH, logger_name="TrainLogger")

# --- FeatureTransformer 类 ---
class FeatureTransformer:
    def __init__(self, discrete_cols=None, continuous_cols=None, invariant_cols=None, model_discrete_cols=None, offset=0):
        """
        初始化 FeatureTransformer.
        :param discrete_cols: list, 离散特征的列名列表.
        :param continuous_cols: list, 连续特征的列名列表.
        :param invariant_cols: list, 不变特征的列名列表 (这些特征在转换过程中保持不变).
        """
        self.discrete_cols = discrete_cols if discrete_cols is not None else [] # 离散特征列
        self.continuous_cols = continuous_cols if continuous_cols is not None else [] # 连续特征列
        self.invariant_cols = invariant_cols if invariant_cols is not None else [] # 不变特征列
        self.model_discrete_cols = model_discrete_cols if model_discrete_cols is not None else []
        self.offset = offset
        
        self.params = {
            'discrete_mappings': {}, # 存储离散特征的映射关系 (key2id, id2key)
            'continuous_stats': {}   # 存储连续特征的统计数据 (均值, 标准差)
        }
        self.fitted = False # 标记 Transformer 是否已经拟合过数据

    def _convert_to_native_python_types(self, data):
        """
        递归地将 numpy 类型转换为原生 Python 类型，以便进行 JSON 序列化。
        JSON 库默认不支持直接序列化 numpy 的特定数值类型。
        """
        if isinstance(data, dict):
            return {k: self._convert_to_native_python_types(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [self._convert_to_native_python_types(i) for i in data]
        elif isinstance(data, (np.integer, np.int_, np.intc, np.intp, np.int8, np.int16, np.int32, np.int64)):
            return int(data) # 将 numpy 整数类型转为 Python int
        elif isinstance(data, (np.floating, np.float_, np.float16, np.float32, np.float64)):
            return float(data) # 将 numpy 浮点数类型转为 Python float
        elif isinstance(data, np.ndarray):
            return data.tolist() # 将 numpy 数组转为 Python list
        return data

    def fit(self, df: pd.DataFrame):
        """
        根据输入的 DataFrame 拟合 Transformer，计算离散特征的映射和连续特征的统计量。
        :param df: pd.DataFrame, 用于拟合的原始特征数据。
        :return: self, 返回拟合后的 Transformer 实例。
        """
        logger.info("开始拟合 FeatureTransformer...")
        # 创建 DataFrame 的副本以避免 SettingWithCopyWarning (如果 df 是一个切片)
        df_copy = df.copy()

        # 处理离散特征
        for col in self.discrete_cols:
            if col not in df_copy.columns:
                logger.warning(f"离散特征列 '{col}' 在 DataFrame 中未找到。跳过。")
                continue
            unique_values = sorted(list(df_copy[col].dropna().unique())) # 获取唯一值并排序以保证映射的一致性
            key2id = {val: i + self.offset for i, val in enumerate(unique_values)} # 创建 值到ID 的映射
            id2key = {i: val for val, i in key2id.items()} # 创建 ID到值 的映射
            self.params['discrete_mappings'][col] = {
                'key2id': key2id,
                'id2key': id2key
            }
            logger.info(f"已拟合离散特征列 '{col}'。映射了 {len(unique_values)} 个唯一值。")

        # 处理连续特征
        for col in self.continuous_cols:
            if col not in df_copy.columns:
                logger.warning(f"连续特征列 '{col}' 在 DataFrame 中未找到。跳过。")
                continue
            mean = float(df_copy[col].mean()) # 计算均值
            std = float(df_copy[col].std())   # 计算标准差
            
            # 处理标准差为0的情况 (例如，列中所有值都相同)
            # if std == 0:
            #     logger.warning(f"连续特征列 '{col}' 的标准差为 0。将使用 std=1.0 以避免除以零错误。")
            #     std = 1.0
            
            self.params['continuous_stats'][col] = {
                'mean': mean,
                'std': std
            }
            logger.info(f"已拟合连续特征列 '{col}'。均值: {mean:.4f}, 标准差: {std:.4f}")
        
        self.fitted = True # 标记为已拟合
        logger.info("FeatureTransformer 拟合完成。")
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        使用拟合好的参数转换 DataFrame 中的特征。
        :param df: pd.DataFrame, 需要转换的特征数据。
        :return: pd.DataFrame, 转换后的特征数据。
        """
        if not self.fitted:
            raise RuntimeError("Transformer 尚未拟合。请先调用 fit() 或 load_params()。")
        
        logger.info("开始转换特征...")
        transformed_df = df.copy() # 创建副本进行转换

        # 转换离散特征
        for col, mappings in self.params['discrete_mappings'].items():
            if col not in transformed_df.columns:
                logger.warning(f"用于转换的离散特征列 '{col}' 在 DataFrame 中未找到。跳过。")
                continue
            key2id = mappings['key2id']
            # 使用 map 函数应用映射。对于在拟合阶段未见过的值，map会产生 NaN。
            transformed_df[col] = transformed_df[col].map(key2id)
            # 检查是否有 NaN 值，这表示在转换（预测）时出现了训练时未见过的值
            if transformed_df[col].isnull().any():
                num_unseen = transformed_df[col].isnull().sum()
                logger.warning(f"列 '{col}' 在转换过程中有 {num_unseen} 个未曾见过的的值。将它们映射为 0。")
                transformed_df[col] = transformed_df[col].fillna(0).astype(int) # 示例：将未见过的的值映射为 0
            logger.info(f"已转换离散特征列 '{col}'。")

        # 转换连续特征 (标准化)
        for col, stats in self.params['continuous_stats'].items():
            if col not in transformed_df.columns:
                logger.warning(f"用于转换的连续特征列 '{col}' 在 DataFrame 中未找到。跳过。")
                continue
            mean = stats['mean']
            std = stats['std']
            # (x - mean) / std 实现标准化
            # transformed_df[col] = (transformed_df[col] - mean) / (std if std != 0 else 1.0) # 避免除以零
            transformed_df[col] = transformed_df[col].apply(lambda x: (x - mean) / std if not np.isnan(x) else np.nan) 
            logger.info(f"已转换连续特征列 '{col}'。")
            
        # 不变特征保持原样 (已经在 transformed_df = df.copy() 时复制)
        for col in self.invariant_cols:
             if col not in transformed_df.columns:
                logger.warning(f"不变特征列 '{col}' 在 DataFrame 中未找到。跳过。")
                continue
             logger.info(f"保持不变特征列 '{col}' 原样。")
        
        logger.info("特征转换完成。")
        return transformed_df

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        先拟合 Transformer，然后转换 DataFrame。
        :param df: pd.DataFrame, 原始特征数据。
        :return: pd.DataFrame, 转换后的特征数据。
        """
        self.fit(df)
        return self.transform(df)
    
    def discrete_column_class_count(self, data: pd.DataFrame) -> dict:
        """
        计算离散列的类别数量。
        :param data: pd.DataFrame, 输入数据。
        :return: dict, 每个离散列的类别数量。
        """
        class_count = {}
        for col in self.model_discrete_cols:
            if col in data.columns:
                # 避免某列全为nan造成embedding初始化失败
                non_null_data = data[col].dropna()
                if len(non_null_data) == 0:
                    logger.warning(f"离散列 '{col}' 无有效数据（全为 NaN）。默认设置为 1 个类别。")
                    class_count[col] = 1  # 避免 Embedding 初始化失败
                    continue

                unique_count = non_null_data.nunique()
                max_val = non_null_data.max()
                if max_val + 1 != unique_count:
                    logger.warning(f"离散列 '{col}' 的编码不连续！最大索引: {max_val}, 实际类别数: {unique_count}")
                class_count[col] = int(max(max_val + 1, unique_count))  # 取较大者
            else:
                logger.warning(f"离散特征列 '{col}' 在 DataFrame 中未找到。跳过。")

        return class_count

    def save_params(self, filepath: Path):
        """
        将拟合好的参数 (映射关系、统计量) 保存到 JSON 文件。
        :param filepath: str, 参数文件的保存路径。
        """
        if not self.fitted:
            logger.warning("Transformer 尚未拟合。正在保存空的或不完整的参数。")
        
        # 在保存前，将 numpy 类型转换为原生 Python 类型
        params_to_save = self._convert_to_native_python_types(self.params)

        os.makedirs(os.path.dirname(filepath), exist_ok=True) #确保目录存在
        with open(filepath, 'w') as f:
            json.dump(params_to_save, f, indent=4) # indent=4 使 JSON 文件更易读
        logger.info(f"Transformer 参数已保存至 {filepath}")

    def load_params(self, filepath: Path):
        """
        从 JSON 文件加载参数。
        :param filepath: str, 参数文件的路径。
        :return: self, 返回加载参数后的 Transformer 实例。
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Transformer 参数文件未找到: {filepath}")
        with open(filepath, 'r') as f:
            self.params = json.load(f)
        
        # 从 JSON 加载后，字典的键（特别是 id2key 中的整数键）会变成字符串。
        # 这里需要将它们转换回整数类型。
        for col_name, mappings in self.params.get('discrete_mappings', {}).items():
            if 'id2key' in mappings and isinstance(mappings['id2key'], dict):
                mappings['id2key'] = {int(k): v for k, v in mappings['id2key'].items()}
        
        self.fitted = True # 假设如果参数被加载，则 Transformer 是已拟合的
        logger.info(f"Transformer 参数已从 {filepath} 加载。")
        return self
# --- FeatureTransformer 类结束 ---

# def train_model(...): # 你的 train_model 函数
#   ... 

if __name__ == "__main__":
    logger.info("开始数据加载和预处理...")
    
    # --- 定义特征和 Transformer 的配置 ---
    # 这部分理想情况下应该来自配置文件或专门的配置对象
    FEATURE_CONFIG = {
        'discrete_cols': ['Month', 'DayOfWeek'], # 示例离散列
        'continuous_cols': ['Temperature', 'Humidity', 'PreviousSales'], # 示例连续列
        'invariant_cols': ['PigFarmID'], # 示例不变列 (通常是ID或静态属性)
        'transformer_path': os.path.join('configs', 'feature_transformer_params.json') # 保存/加载参数的路径
    }
    # 确保 Transformer 参数文件所在的目录存在
    os.makedirs(os.path.dirname(FEATURE_CONFIG['transformer_path']), exist_ok=True)

    # 1. 加载和基础预处理数据
    feature_generator = FeatureGenerator() # 实例化特征生成器
    feature_df = feature_generator.generate_features() # 生成特征 DataFrame

    if feature_df is None or feature_df.empty:
        logger.error("特征数据加载失败或为空，程序退出。")
        exit()
    
    logger.info(f"原始特征数据 (前5行):\n{feature_df.head()}")

    # 2. 初始化并使用 FeatureTransformer
    transformer = FeatureTransformer(
        discrete_cols=FEATURE_CONFIG['discrete_cols'],
        continuous_cols=FEATURE_CONFIG['continuous_cols'],
        invariant_cols=FEATURE_CONFIG['invariant_cols']
    )

    # --- 训练模式：拟合和转换，然后保存参数 ---
    # transformed_feature_df = transformer.fit_transform(feature_df) # 拟合并转换
    # transformer.save_params(FEATURE_CONFIG['transformer_path']) # 保存参数
    
    # --- 或者：预测/推理模式：加载参数并转换 ---
    # 首先，确保你有一个在训练运行中保存的 'feature_transformer_params.json' 文件。
    # 为了演示，我们首先假设处于“训练”模式以创建参数文件。
    
    # 训练模式示例:
    logger.info("当前运行在 'fit_transform' 模式，以生成并保存 Transformer 参数。")
    # 使用 .copy() 如果 feature_df 之后还会被使用，以避免修改原始 DataFrame
    transformed_feature_df = transformer.fit_transform(feature_df.copy()) 
    transformer.save_params(FEATURE_CONFIG['transformer_path'])
    logger.info(f"转换后的特征数据 (前5行):\n{transformed_feature_df.head()}")

    # 演示加载参数（例如，在预测脚本或单独的运行中）:
    logger.info("\n--- 模拟预测模式：加载 Transformer 参数 ---")
    transformer_for_prediction = FeatureTransformer( # 使用相同的列配置进行初始化
        discrete_cols=FEATURE_CONFIG['discrete_cols'],
        continuous_cols=FEATURE_CONFIG['continuous_cols'],
        invariant_cols=FEATURE_CONFIG['invariant_cols']
    )
    try:
        transformer_for_prediction.load_params(FEATURE_CONFIG['transformer_path'])
        # 现在，如果你有新的数据 (例如 new_feature_df)，你可以转换它:
        # new_transformed_df = transformer_for_prediction.transform(new_feature_df.copy())
        # logger.info(f"使用加载的参数转换新数据 (用原始数据为例):\n{transformer_for_prediction.transform(feature_df.copy()).head()}")
        # 为了演示，让我们重新转换原始 df 以显示加载功能正常工作
        re_transformed_df = transformer_for_prediction.transform(feature_df.copy())
        logger.info(f"使用加载的参数重新转换的数据 (前5行):\n{re_transformed_df.head()}")
        
        # 验证 re_transformed_df 是否与 transformed_feature_df 非常接近
        # (考虑到可能的浮点不精确性)
        # 对于离散和不变列，它们应该是相同的。
        # 对于连续列，它们应该几乎相同。
        # pd.testing.assert_frame_equal(transformed_feature_df, re_transformed_df, check_dtype=False, atol=1e-5)
        # logger.info("验证：重新转换的数据与原始转换的数据匹配。")

    except FileNotFoundError as e:
        logger.error(f"无法加载 Transformer 参数进行预测: {e}")
        logger.error("请首先在训练模式下运行以生成参数文件。")

    # 用于你的流程其余部分的实际数据的占位符
    # 此时, `transformed_feature_df` (来自 fit_transform) 
    # 或 `re_transformed_df` (来自 load_params 然后 transform)
    # 是你将用于创建序列、数据集等的 DataFrame。
    # 例如，如果继续训练流程:
    current_features_to_use = transformed_feature_df 


    # 2. 划分训练集和验证集 (示例：按猪场ID划分)
    # ... (你现有的 split_data_by_farm_id 逻辑将使用 `current_features_to_use`)
    # train_df, val_df, _ = split_data_by_farm_id(
    # current_features_to_use, # 使用转换后的数据
    #     config.PIG_FARM_ID_COLUMN,
    #     config.VALIDATION_SPLIT_RATIO,
    #     test_ratio=0, 
    #     random_seed=config.RANDOM_SEED
    # )
    # logger.info(f"训练集猪场数: {train_df[config.PIG_FARM_ID_COLUMN].nunique()}, 验证集猪场数: {val_df[config.PIG_FARM_ID_COLUMN].nunique()}")

    # 3. 为训练集和验证集创建序列
    # X_train, y_train = create_sequences(...) # 这现在将使用转换后的特征
    # ... 你的流程的其余部分 ...


    # --- 你的主脚本的其余部分 ---
    # ... (DataLoaders, Model, Training loop, etc.) ...
    logger.info("特征转换部分演示完毕。")
    logger.info("请将 'current_features_to_use' 集成到你的数据拆分和序列创建步骤中。")
    logger.info("请取消注释并实现 LSTM 模型部分以及后续的数据处理流程以开始训练。")