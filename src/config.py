# config.py
import torch
from pathlib import Path
# --- 数据目录 ---
ROOT_DIR = Path(__file__).parent.parent
DATA_DIR = ROOT_DIR / "data"
MODELS_DIR = ROOT_DIR / "models"

# --- 日期 ---
TRAIN_RUNNING_DT = "2024-3-1" # 运行日期 (用于数据切分)
TRAIN_INTERVAL = 100 # 训练数据的时间间隔 (单位：天)

# --- 数据相关配置 ---
VALIDATION_SPLIT_RATIO = 0.2 # 验证集占总猪场数据的比例 (按猪场ID划分时)
# 或者 VALIDATION_CUTOFF_DATE = '2022-01-01' # 按时间划分验证集
TEST_SPLIT_RATIO = 0.2 # 测试集比例 (如果需要的话)

TRANSFORM_OFFSET = 1

# --- 训练相关配置 ---
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = 4096
NUM_EPOCHS = 10
LEARNING_RATE = 0.001

L2_REGULARIZATION = 1e-4 # L2 正则化系数

# --- 模型相关配置 (LSTM 模型参数，先占位) ---
DROPOUT = 0.2

EMBEDDING_SIZE = 128 # 嵌入层大小 (如果使用嵌入层)
NUM_WORKERS = 16

# --- 其他配置 ---
MODEL_SAVE_PATH = MODELS_DIR / "model.pth" # 模型保存路径
TRANSFORMER_SAVE_PATH = MODELS_DIR / "transformer.json" # 特征转换器保存路径
RANDOM_SEED = 42 # 随机种子，保证结果可复现

class main_predict:

    PREDICT_RUNNING_DT = "2024-4-21" # 运行日期 (用于数据切分)
    PREDICT_INTERVAL = 50 # 训练数据的时间间隔 (单位：天)


    PREDICT_DATA_DIR = DATA_DIR / "predict"

    HAS_RISK_PREDICT_RESULT_SAVE_PATH = PREDICT_DATA_DIR / "has_risk_predict_result.csv"
    DAYS_PREDICT_RESULT_SAVE_PATH = PREDICT_DATA_DIR / "days_predict_result.csv"

    PREDICT_INDEX_TABLE = PREDICT_DATA_DIR / "index_sample_20240301.csv"

