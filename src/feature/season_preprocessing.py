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
        
        # 计算季节
        self.index_data['season'] = ((self.index_data['month'] - 1) // 3 + 1).astype(int)
        
        # 映射季节到字符串
        # season_map = {0: '春', 1: '夏', 2: '秋', 3: '冬'}
        # self.index_data['season'] = self.index_data['season'].map(season_map)
        
        return self.index_data