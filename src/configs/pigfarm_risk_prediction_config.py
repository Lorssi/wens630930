from utils import os_util
import configs.base_config as base_config
from enum import Enum

from pathlib import Path

# 获取项目根目录（src的父目录）
ROOT_DIR = Path(__file__).parent.parent.parent
# 数据目录
DATA_DIR = ROOT_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw_data"
INTERIM_DATA_DIR = DATA_DIR / "interim_data"
FEATURE_STORE_DIR = INTERIM_DATA_DIR / "feature_store"

algo_interim_dir = INTERIM_DATA_DIR / "risk_prediction"

# algo_interim_dir = os_util.create_dir_if_not_exist('/'.join([base_config.INTERIM_DATA_ROOT, 'weight_prediction']))
# algo_model_dir = os_util.create_dir_if_not_exist('/'.join([base_config.MODEL_DATA_ROOT, 'weight_prediction']))

predict_data_dt_path =  algo_interim_dir / "RiskPredictionNfm.model.predict_dt.csv"
predict_data_dt_path_type = 'csv'


class TrainModuleConfig(Enum):
    # 训练时间间隔
    TRAIN_ALGO_INTERVAL = 100
    # 表现期长度
    OUTCOME_WINDOW_LEN = None
    # 表现期偏移量
    OUTCOME_OFFSET = None

    # 训练参数
    num_iter = 10
    batch_size = 128
    lr = 0.001
    momentum = 0.9
    l2_decay = 1e-2
    seed = 42


class PredictModuleConfig(Enum):
    # 预测时间偏移量
    PREDICT_OUTCOME_OFFSET = None
    # 预测时间间隔
    PREDICT_RUNNING_DT_INTERVAL = 14

    predict_days_interval = (50, 130)


class EvalModuleConfig(Enum):
    # 测试时间偏移量
    OUTCOME_OFFSET = 1
    # 测试时间间隔
    EVAL_INTERVAL = 30
    # 测试时间窗口长度
    OUTCOME_WINDOW_LEN_DT = 3
    OUTCOME_WINDOW_LEN_INDEX = 3


BASE_EMB_DIM = 128
