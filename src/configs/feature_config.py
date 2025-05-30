# DATA
import os
from pathlib import Path

class DataPathConfig:
    # 获取项目根目录（src的父目录）
    ROOT_DIR = Path(__file__).parent.parent.parent
    
    # 数据目录
    DATA_DIR = ROOT_DIR / "data"
    RAW_DATA_DIR = DATA_DIR / "raw_data"
    INTERIM_DATA_DIR = DATA_DIR / "interim_data"
    
    # 确保目录存在
    os.makedirs(RAW_DATA_DIR, exist_ok=True)
    os.makedirs(INTERIM_DATA_DIR, exist_ok=True)
    
    # PATH - 使用相对路径
    ML_DATA_PATH = RAW_DATA_DIR / "ads_pig_org_total_to_ml_training_day  生产数据.csv"
    INDEX_ABORTION_RATE_DATA_SAVE_PATH = INTERIM_DATA_DIR / "ads_pig_org_total_to_ml_training_day_abortion.csv"
    ML_AOBRTION_RATE_DATA_SAVE_PATH = INTERIM_DATA_DIR / "ml_abortion_rate.csv"

    ABORTION_LABEL_DATA_SAVE_PATH = INTERIM_DATA_DIR / "abortion_label.csv"
    NON_DROP_NAN_LABEL_DATA_SAVE_PATH = INTERIM_DATA_DIR / "non_drop_nan_label.csv"
    ABORTION_CALCULATE_DATA_SAVE_PATH = INTERIM_DATA_DIR / "abortion_calculate_data_label.csv"

    W01_AST_BOAR_PATH = RAW_DATA_DIR / "W01_AST_BOAR.csv"
    TMP_ADS_PIG_ISOLATION_TAME_RISK_L1_N2 = RAW_DATA_DIR / "TMP_ADS_PIG_ISOLATION_TAME_RISK_L1_N2.csv"
    INTRO_DATA_SAVE_PATH = INTERIM_DATA_DIR / "intro_data.csv"

    DIM_ORG_INV_PATH = RAW_DATA_DIR / "dim_org_inv.csv"
    ORG_FEATURE_DATA_SAVE_PATH = INTERIM_DATA_DIR / "org_feature_data.csv"

    PRRS_CHECK_DATA_PATH = RAW_DATA_DIR / "TMP_PIG_ORG_DISEASE_CHECK_RESULT_DAY  检测数据猪场.csv"
    PRRS_PICK_DATA_SAVE_PATH = INTERIM_DATA_DIR / "prrs_pick_check_data.csv"

    DEATH_CONFIRM_DATA_PATH = RAW_DATA_DIR / "TMP_ORG_PRRS_OVERALL_ADOPT_CULL_DAY.csv"

    FEATURE_DATA_SAVE_PATH = INTERIM_DATA_DIR / "feature_data.csv"
    TRAIN_TRANSFORMED_FEATURE_DATA_SAVE_PATH = INTERIM_DATA_DIR / "train_transformed_feature_data.csv"
    VAL_TRANSFORMED_FEATURE_DATA_SAVE_PATH = INTERIM_DATA_DIR / "val_transformed_feature_data.csv"

    DATA_TRANSFORMED_MASK_NULL_TRAIN_PATH = INTERIM_DATA_DIR / "transformed_feature_data_mask_null_train.csv"
    DATA_TRANSFORMED_MASK_NULL_VAL_PATH = INTERIM_DATA_DIR / "transformed_feature_data_mask_null_val.csv"
    DATA_TRANSFORMED_MASK_NULL_PREDICT_PATH = INTERIM_DATA_DIR / "transformed_feature_data_mask_null_predict.csv"

    PREDICT_DATA_TRANSFORMED_MASK_NULL_PREDICT_PATH = INTERIM_DATA_DIR / "predict_df.csv"
    
    # 返回字符串路径的方法
    @classmethod
    def get_path(cls, path_attr):
        """获取路径字符串"""
        path = getattr(cls, path_attr)
        if isinstance(path, Path):
            return str(path)
        return path

class ColumnsConfig:
    INDEX_DATA_COLUMN = ['stats_dt', 'pigfarm_dk', 'city', 'abortion_rate']
    MAIN_PREDICT_DATA_COLUMN = ['stats_dt', 'pigfarm_dk', 'abort_1_7', 'abort_8_14', 'abort_15_21', 'abort_1_7_pred', 'abort_8_14_pred', 'abort_15_21_pred',
                                'abort_1_7_decision', 'abort_8_14_decision', 'abort_15_21_decision',
                                'abort_1_7_threshold', 'abort_8_14_threshold', 'abort_15_21_threshold']
    DAYS_PREDICT_DATA_COLUMN = ['stats_dt', 'pigfarm_dk', 'days_1_7', 'days_8_14', 'days_15_21', 'abort_day_1_7_pred', 'abort_day_8_14_pred', 'abort_day_15_21_pred',
                                'days_1_7_decision', 'days_8_14_decision', 'days_15_21_decision',
                                'days_1_7_threshold', 'days_8_14_threshold', 'days_15_21_threshold']

    # TRANSFORM_FIT
    DISCRETE_COLUMNS = ['pigfarm_dk','city']
    CONTINUOUS_COLUMNS = ['abortion_rate_1_7','check_out_ratio_7d', 'death_confirm_2_week','intro_source_num_90d', 'intro_times_90d', 'boar_tame_num_30d']
    INVARIANT_COLUMNS = ['season', 'intro_source_is_single']

    # MODEL_FIT
    MODEL_DISCRETE_COLUMNS = ['pigfarm_dk','city','season', 'intro_source_is_single']
    MODEL_CONTINUOUS_COLUMNS = ['abortion_rate_1_7', 'check_out_ratio_7d', 'death_confirm_2_week','intro_source_num_90d', 'intro_times_90d', 'boar_tame_num_30d']
    MODEL_DIRECT_COLUMNS = []

    # COLUMNS
    feature_columns = MODEL_DISCRETE_COLUMNS + MODEL_DIRECT_COLUMNS + MODEL_CONTINUOUS_COLUMNS

    # label
    HAS_RISK_LABEL = 'has_risk_label'

    HAS_RISK_4_CLASS_PRE = 'abort_{}_{}'
    DAYS_RISK_8_CLASS_PRE = 'days_{}_{}'

    HAS_RISK_3_POINT_LABEL = ["future_7_label", "future_14_label", "future_21_label"]