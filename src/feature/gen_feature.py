from feature.ml_preprocessing import MLDataPreprocessor
from feature.intro_preprocessing import IntroDataPreprocessor
from feature.season_preprocessing import SeasonPreprocessor
from feature.dim_org import OrgDataPreprocessor
from feature.surrounding_preprocessing import SurroundingPreprocessing
from feature.check_preprocessing import CheckPreprocessor
from feature.death_confirm_preprocessing import DeathConfirmPreprocessor
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
        intro_data = IntroDataPreprocessor(DataPathConfig.W01_AST_BOAR_PATH, 
                                           DataPathConfig.TMP_ADS_PIG_ISOLATION_TAME_RISK_L1_N2, 
                                           index_data=index_data)
        if intro_data.intro_data is None:
            logger.error("数据加载失败，无法计算引种数据特征")
            return

        # 计算引种特征
        intro_feature = intro_data.calculate_intro_feature()
        if intro_feature is None:
            logger.error("引种特征计算失败")

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
        season_feature = season_data.calculate_season()
        logger.info("季节特征计算完成")
        
        return season_feature
    
    def dim_org_feature(self, index_data=None):
        """
        计算组织数据特征
        """
        # 加载组织数据
        org_data = OrgDataPreprocessor(DataPathConfig.DIM_ORG_INV_PATH)
        if org_data.org_data is None:
            logger.error("组织数据加载失败，无法计算组织数据特征")
            return

        # 合并组织数据
        org_feature = org_data.get_dim_org_data(index_data)
        logger.info("组织数据特征计算完成")
        
        return org_feature
    
    def surrounding_feature(self, index_data=None):
        """
        计算周边信息特征
        """
        # 加载周边信息数据
        surrounding_data = SurroundingPreprocessing(index_data=index_data)
        surrounding_data.calculate_surrounding_feature()
        if surrounding_data.index_data is None:
            logger.error("周边信息数据加载失败，无法计算周边信息特征")
            return
        return surrounding_data.index_data
    
    def prrs_check_feature(self, index_data=None):
        """
        计算PRRS检查特征
        """
        checkPreprocessor = CheckPreprocessor(index_data=index_data, running_dt=self.running_dt, interval_days=self.interval_days)
        check_data = checkPreprocessor.calculate_check_out_ratio()
        return check_data

    def production_feature(self, index_data=None):
        """
        计算生产数据特征
        """
        pass

    def death_confirm_feature(self, index_data=None):
        """
        计算死亡确认数据特征
        """
        death_confirm_data = DeathConfirmPreprocessor(index_data=index_data, running_dt=self.running_dt, interval_days=self.interval_days, death_confirm_data_path=DataPathConfig.DEATH_CONFIRM_DATA_PATH)
        death_confirm_feature = death_confirm_data.calculate_death_confirm_feature()
        return death_confirm_feature


    def generate_features(self):
        """
        生成特征
        """
        if self.abortion_data is None:
            logger.error("流产率数据未加载，无法生成特征")
            return
        feature = self.abortion_data.copy()
        feature = self.dim_org_feature(feature)
        # feature = self.intro_data_feature(self.feature)
        feature = self.season_feature(feature)
        feature = self.surrounding_feature(feature)
        feature = self.prrs_check_feature(feature)
        feature = self.death_confirm_feature(feature)

        feature = feature[['stats_dt'] + ColumnsConfig.feature_columns]
        feature.to_csv(DataPathConfig.FEATURE_DATA_SAVE_PATH, index=False, encoding='utf-8')
        # 这里可以添加更多的特征生成逻辑
        # 例如：self.abortion_data['new_feature'] = self.abortion_data['abortion_rate'] * 2
        logger.info(f"特征生成完成,特征数据保存至 {DataPathConfig.FEATURE_DATA_SAVE_PATH}")
        return feature
