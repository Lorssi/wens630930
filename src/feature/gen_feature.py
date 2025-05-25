from feature.ml_preprocessing import MLDataPreprocessor
from feature.intro_preprocessing import IntroDataPreprocessor
from feature.season_preprocessing import SeasonPreprocessor
from configs.feature_config import DataPathConfig, ColumnsConfig
from configs.logger_config import logger_config
import pandas as pd

from utils.logger import setup_logger
logger = setup_logger(logger_config.TRAIN_LOG_FILE_PATH, logger_name="FeatureLogger")

class FeatureGenerator:
    def __init__(self, running_dt = "2024-10-01", interval_days = 400):
        self.running_dt = pd.to_datetime(running_dt)
        self.interval_days = interval_days

        self.abortion_data = self.abortion_rate_feature()

    def abortion_rate_feature(self):
        """
        计算流产率特征
        """
        # 加载数据
        ml_data = MLDataPreprocessor(DataPathConfig.ML_DATA_PATH, running_dt=self.running_dt, interval_days=self.interval_days)
        if ml_data.index_data is None:
            logger.error("数据加载失败，无法计算流产率特征")
            return

        # 计算流产率
        ml_data.calculate_abortion_rate()
        ml_data.clean_data()
        ml_data.clean_ml_data()
    
        return ml_data.index_data
    
    def intro_data_feature(self, index_data=None):
        """
        计算引种数据特征
        """
        # 加载数据
        intro_data = IntroDataPreprocessor(DataPathConfig.W01_AST_BOAR_PATH, index_data=index_data, running_dt=self.running_dt, interval_days=self.interval_days)
        if intro_data.intro_data is None:
            logger.error("数据加载失败，无法计算引种数据特征")
            return

        # 计算引种特征
        intro_feature = intro_data.calculate_is_single_and_intro_num()
        
        # 保存数据
        intro_feature.to_csv(DataPathConfig.INTRO_DATA_SAVE_PATH, index=False, encoding='utf-8')
        logger.info(f"引种数据特征保存至 {DataPathConfig.INTRO_DATA_SAVE_PATH}")
        return intro_feature

    def season_feature(self, index_data=None):
        """
        计算季节特征
        """
        # 加载数据
        season_data = SeasonPreprocessor(index_data=index_data)
        if season_data.index_data is None:
            logger.error("数据加载失败，无法计算季节特征")
            return

        # 计算季节特征
        season_feature = season_data.calculate_month()
        logger.info("季节特征计算完成")
        
        return season_feature

    def generate_features(self):
        """
        生成特征
        """
        if self.abortion_data is None:
            logger.error("流产率数据未加载，无法生成特征")
            return
        feature = self.intro_data_feature(self.abortion_data)
        feature = self.season_feature(feature)

        feature = feature[['stats_dt'] + ColumnsConfig.feature_columns]
        feature.to_csv(DataPathConfig.FEATURE_DATA_SAVE_PATH, index=False, encoding='utf-8')
        # 这里可以添加更多的特征生成逻辑
        # 例如：self.abortion_data['new_feature'] = self.abortion_data['abortion_rate'] * 2
        logger.info(f"特征生成完成,特征数据保存至 {DataPathConfig.FEATURE_DATA_SAVE_PATH}")
        return feature
