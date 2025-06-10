from enum import Enum
from utils import os_util
from pathlib import Path

# 获取项目根目录（src的父目录）
ROOT_DIR = Path(__file__).parent.parent.parent
# 数据目录
DATA_DIR = ROOT_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw_data"
INTERIM_DATA_DIR = DATA_DIR / "interim_data"
FEATURE_STORE_DIR = INTERIM_DATA_DIR / "feature_store"

class RawData(Enum):
    
    # 生产数据表
    ADS_PIG_ORG_TOTAL_TO_ML_TRAINING_DAY = RAW_DATA_DIR / "ads_pig_org_total_to_ml_training_day  生产数据.csv"
    # 组织维度表
    DIM_ORG_INV = RAW_DATA_DIR / "dim_org_inv.csv"
    # 死淘表
    TMP_ORG_PRRS_OVERALL_ADOPT_CULL_DAY = RAW_DATA_DIR / "TMP_ORG_PRRS_OVERALL_ADOPT_CULL_DAY.csv"
    # 检测表
    TMP_PIG_ORG_DISEASE_CHECK_RESULT_DAY = RAW_DATA_DIR / "TMP_PIG_ORG_DISEASE_CHECK_RESULT_DAY  检测数据猪场.csv"
    # 引种数据
    W01_AST_BOAR = RAW_DATA_DIR / "W01_AST_BOAR.csv"
    TMP_ADS_PIG_ISOLATION_TAME_RISK_L1_N2 = RAW_DATA_DIR / "TMP_ADS_PIG_ISOLATION_TAME_RISK_L1_N2.csv"
    ADS_PIG_ISOLATION_TAME_PROLINE_RISK = RAW_DATA_DIR / "ADS_PIG_ISOLATION_TAME_PROLINE_RISK.csv"

    PIG_LET_DATA = RAW_DATA_DIR / "ads_pig_efficient_piglet_batch_analysis_day.csv"

    ABNORMAL_BOAR_REPORT_MODEL_DATA = RAW_DATA_DIR / "ADS_ABNORMAL_BOAR_REPORT_MODEL.csv"

class FeatureData(Enum):

    PRODUCTION_FEATURE_DATA = FEATURE_STORE_DIR / "production_feature_data.csv"

    ORG_FEATURE_DATA = FEATURE_STORE_DIR / "org_feature_data.csv"

    DATE_FEATURE_DATA = FEATURE_STORE_DIR / "date_feature_data.csv"

    CHECK_FEATURE_DATA = FEATURE_STORE_DIR / "check_feature_data.csv"

    DEATH_CONFIRM_FEATURE_DATA = FEATURE_STORE_DIR / "death_confirm_feature_data.csv"

    INTRO_FEATURE_DATA = FEATURE_STORE_DIR / "intro_feature_data.csv"

    SORROUNDING_FEATURE_DATA = FEATURE_STORE_DIR / "sorrounding_feature_data.csv"

    RULE_BASELINE_FEATURE_DATA = FEATURE_STORE_DIR / "rule_baseline_feature_data.csv"

    ABNORMAL_BOAR_FEATURE_DATA = FEATURE_STORE_DIR / "abnormal_boar_feature_data.csv"


class ModulePath(Enum):
    # 数据预处理模块
    pre_process_data_class = 'PreProcessDataset'
    # 特征模块
    feature_dataset_list = [
        # 生产特征-7天
        {'dataset_name': 'dataset.production_feature',
         'file_type': 'csv',
         'main_class_name': 'ProductionFeature',
         'params': {}},
        # 组织特征
        {'dataset_name': 'dataset.org_location_feature',
         'file_type': 'csv',
         'main_class_name': 'OrgLocationFeature',
         'params': {}},
        # 日期特征
        {'dataset_name': 'dataset.date_feature',
         'file_type': 'csv',
         'main_class_name': 'DateFeature',
         'params': {}},
        # 检测特征
        {'dataset_name': 'dataset.prrs_check_feature',
         'file_type': 'csv',
         'main_class_name': 'PrrsCheckFeature',
         'params': {}},
        # 死淘特征
        {'dataset_name': 'dataset.death_confirm_feature',
         'file_type': 'csv',
         'main_class_name': 'DeathConfirmFeature',
         'params': {}},
        # # 引种特征
        # {'dataset_name': 'dataset.intro_feature',
        #  'file_type': 'csv',
        #  'main_class_name': 'IntroFeature',
        #  'params': {}},
        # {'dataset_name': 'dataset.surrounding_pigfarmInfo_feature',
        #  'file_type': 'csv',
        #  'main_class_name': 'SurroundingPigfarmInfoFeature',
        #  'params': {}},
        {'dataset_name': 'dataset.rule_baseline_feature',
         'file_type': 'csv',
         'main_class_name': 'RulBaselineFeature',
         'params': {}},
        {'dataset_name': 'dataset.abnormal_breed_sow_feature',
         'file_type': 'csv',
         'main_class_name': 'AbnormalBreedSowFeature',
         'params': {}},
    ]
    # 训练模块
    algo_list = [
        # 任务1 均重预测
        {'algo_name': 'algorithm.weight_prediction_nfm_algo',
         'algo_main_class': 'WeightPredictionNfmAlgo',
         'algo_config': 'config.weight_prediction_config',
         'algo_simple_name': 'WP'},
    ]
    # 预测模块
    algo_predict = [
        # 任务1-均重预测
        {'algo_name': 'algorithm.weight_prediction_nfm_algo',
         'algo_main_class': 'WeightPredictionNfmAlgo',
         'algo_config': 'config.weight_prediction_config',
         'algo_simple_name': 'WP'},
    ]

    # 测试模块
    eval_list = [
        # 任务1-均重预测
        {'predict_class_name': 'algorithm.weight_prediction_nfm_algo',
         'predict_main_class_name': 'WeightPredictionNfmAlgo',
         'eval_class_name': 'eval.main',
         'eval_main_class_name': 'AvgWtEval',
         'alog_config_name': 'config.weight_prediction_config',
         'params': {}},
    ]


