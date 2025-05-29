# data_loader/preprocessing.py
import pandas as pd
import numpy as np
from utils.logger import setup_logger
from configs.logger_config import logger_config
logger = setup_logger(logger_config.TRAIN_LOG_FILE_PATH, logger_name="PreprocessingLogger")

class SeasonPreprocessor:
    def __init__(self, index_data):
        self.index_data = index_data.copy()  # 深拷贝，避免对原数据的修改

    def calculate_month(self):
        """
        计算月份特征
        """
        # 确保日期列是 datetime 类型
        self.index_data['stats_dt'] = pd.to_datetime(self.index_data['stats_dt'])
        
        # 提取月份
        self.index_data['month'] = self.index_data['stats_dt'].dt.month
        
        return self.index_data

    def calculate_season(self):
        """
        计算季节特征
        """
        # 确保日期列是 datetime 类型
        self.index_data['stats_dt'] = pd.to_datetime(self.index_data['stats_dt'])
        
        # 提取月份
        self.index_data['month'] = self.index_data['stats_dt'].dt.month
        
        # 气象学季节划分
        season_map = {
            1: 4,  # 1月 -> 冬季
            2: 4,  # 2月 -> 冬季
            3: 1,  # 3月 -> 春季
            4: 1,  # 4月 -> 春季
            5: 1,  # 5月 -> 春季
            6: 2,  # 6月 -> 夏季
            7: 2,  # 7月 -> 夏季
            8: 2,  # 8月 -> 夏季
            9: 3,  # 9月 -> 秋季
            10: 3, # 10月 -> 秋季
            11: 3, # 11月 -> 秋季
            12: 4  # 12月 -> 冬季
        }
        self.index_data['season'] = self.index_data['month'].map(season_map).astype(int)
        
        return self.index_data