import pandas as pd
import numpy as np
from utils.logger import setup_logger
from configs.logger_config import logger_config
from tqdm import tqdm
logger = setup_logger(logger_config.TRAIN_LOG_FILE_PATH, logger_name="PreprocessingLogger")

class DeathConfirmPreprocessor:
    def __init__(self, index_data, death_confirm_data_path, running_dt="2024-10-01", interval_days=400):
        self.index_data = index_data.copy()
        self.death_confirm_data = pd.read_csv(death_confirm_data_path, encoding='utf-8')
        
        self.end = pd.to_datetime(running_dt) - pd.Timedelta(days=1)
        self.start = self.end - pd.Timedelta(days=interval_days) - pd.Timedelta(days=36)

    def calculate_death_confirm_feature(self):
        """
        计算动态两周死淘率和动态五周死淘率特征
        """
        if self.death_confirm_data.empty:
            logger.warning("死亡确认数据为空，无法计算特征")
            self.index_data['death_confirm_2_week'] = np.nan
            self.index_data['death_confirm_5_week'] = np.nan
            return self.index_data
        
        # 确保数据类型正确
        self.death_confirm_data['stats_dt'] = pd.to_datetime(self.death_confirm_data['stats_dt'])
        self.death_confirm_data['stats_dt'] = self.death_confirm_data['stats_dt'] + pd.Timedelta(days=1)  # 确保日期是正确的
        
        self.death_confirm_data.rename(columns={'org_inv_dk': 'pigfarm_dk'}, inplace=True)
        
        # 筛选时间范围内的数据
        filtered_data = self.death_confirm_data[
            (self.death_confirm_data['stats_dt'] >= self.start) & 
            (self.death_confirm_data['stats_dt'] <= self.end)
        ]
        
        # 如果筛选后没有数据，则返回空值
        if filtered_data.empty:
            logger.warning("筛选后死亡确认数据为空，无法计算特征")
            self.index_data['death_confirm_2_week'] = np.nan
            self.index_data['death_confirm_5_week'] = np.nan
            return self.index_data
        
        # 预处理数据，只保留需要的列，提高处理效率
        death_data_processed = filtered_data[
            ['pigfarm_dk', 'stats_dt', 
             'ago_14_days_adopt_qty', 'ago_14_days_adopt_qty_sum', 'ago_14_days_cull_qty',
             'ago_35_days_adopt_qty', 'ago_35_days_adopt_qty_sum', 'ago_35_days_cull_qty']
        ].copy()
        
        # 确保索引数据的日期是datetime类型
        self.index_data['stats_dt'] = pd.to_datetime(self.index_data['stats_dt'])
        
        # 预先计算分子部分 - 向量化操作提高效率
        death_data_processed['numerator_2_week'] = (
            death_data_processed['ago_14_days_cull_qty'] * 
            death_data_processed['ago_14_days_adopt_qty'] / 
            death_data_processed['ago_14_days_adopt_qty_sum']
        )
        
        death_data_processed['numerator_5_week'] = (
            death_data_processed['ago_35_days_cull_qty'] * 
            death_data_processed['ago_35_days_adopt_qty'] / 
            death_data_processed['ago_35_days_adopt_qty_sum']
        )
        
        # 按猪场和日期分组高效计算死淘率
        death_rate_data = death_data_processed.groupby(['pigfarm_dk', 'stats_dt']).agg({
            'numerator_2_week': 'sum',
            'numerator_5_week': 'sum',
            'ago_14_days_adopt_qty': 'sum',
            'ago_35_days_adopt_qty': 'sum'
        })
        
        # 一次性计算死淘率
        death_rate_data['death_confirm_2_week'] = death_rate_data['numerator_2_week'] / death_rate_data['ago_14_days_adopt_qty']
        death_rate_data['death_confirm_5_week'] = death_rate_data['numerator_5_week'] / death_rate_data['ago_35_days_adopt_qty']
        
        # 处理除以零的情况
        death_rate_data['death_confirm_2_week'] = death_rate_data['death_confirm_2_week'].replace([np.inf, -np.inf], np.nan)
        death_rate_data['death_confirm_5_week'] = death_rate_data['death_confirm_5_week'].replace([np.inf, -np.inf], np.nan)
        
        # 四舍五入到4位小数
        death_rate_data['death_confirm_2_week'] = death_rate_data['death_confirm_2_week'].round(4)
        death_rate_data['death_confirm_5_week'] = death_rate_data['death_confirm_5_week'].round(4)
        
        # 只保留需要的列，减少内存占用
        result_data = death_rate_data[['death_confirm_2_week', 'death_confirm_5_week']].reset_index()
        
        # 将结果合并到index_data
        self.index_data = self.index_data.merge(
            result_data,
            on=['pigfarm_dk', 'stats_dt'],
            how='left'
        )
        
        logger.info(f"计算完成死亡确认两周和五周死淘率，总数据量: {len(self.index_data)}")
        
        return self.index_data