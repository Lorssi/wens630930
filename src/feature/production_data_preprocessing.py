import pandas as pd
import numpy as np

from utils.logger import setup_logger
from configs.logger_config import logger_config
logger = setup_logger(logger_config.TRAIN_LOG_FILE_PATH, logger_name="PreprocessingLogger")


class ProductionDataPreprocessor:
    def __init__(self, data_path, index_data, running_dt, interval_days):
        self.data_path = data_path
        self.index_data = index_data
        self.start_date = pd.to_datetime(running_dt) - pd.Timedelta(days=interval_days) - pd.Timedelta(days=31) # 用于给开始日期留出计算数据
        self.end_date = pd.to_datetime(running_dt) - pd.Timedelta(days=1)
        self.abortion_feature_columns = ['abortion_feature_1_7', 'abortion_mean_recent_7d', 'abortion_mean_recent_14d', 'abortion_mean_recent_21d']
        self.boar_feature_columns = ['boar_transin_times_30d', 'boar_transin_qty_30d', 'boar_transin_ratio_30d_1', 'boar_transin_ratio_30d_2']
        self.preg_stock_feature_columns = ['preg_stock_sqty_change_ratio_7d', 'preg_stock_sqty_change_ratio_15d', 'preg_stock_qty']
        self.reserve_sow_feature_columns = ['reserve_sow_sqty', 'reserve_sow_sqty_change_ratio_7d', 'reserve_sow_sqty_change_ratio_15d']
        self.basesow_feature_columns = ['basesow_sqty', 'basesow_sqty_change_ratio_7d', 'basesow_sqty_change_ratio_15d']
        self.load_data()

    def load_data(self):
        try:
            self.production_data = pd.read_csv(self.data_path)
            self.production_data['stats_dt'] = pd.to_datetime(self.production_data['stats_dt'])
            # 过滤数据，确保日期在指定范围内
            self.production_data = self.production_data[(self.production_data['stats_dt'] >= self.start_date) &
                                                        (self.production_data['stats_dt'] <= self.end_date)]
            # 排序，用于加速
            self.production_data.sort_values(by=['stats_dt', 'pigfarm_dk'], inplace=True)
        except Exception as e:
            logger.error(f"加载生产数据失败: {e}")
            self.production_data = None
    
    
    def calculate_abortion_rate(self):
        # 确保流产数量和怀孕母猪存栏量是数值类型，并将NaN填充为0，因为它们参与计算
        self.production_data['abort_qty'] = pd.to_numeric(self.production_data['abort_qty'], errors='coerce').fillna(0)
        self.production_data['preg_stock_qty'] = pd.to_numeric(self.production_data['preg_stock_qty'], errors='coerce').fillna(0)
        
        # 使用 groupby 和 rolling window 计算每个猪场每个日期的近7天流产总数
        # 'closed="left"' 通常用于rolling sum，但这里我们需要包含当天，所以默认'right'就可以
        # min_periods=1 表示即使不足7天，也会计算已有的天数和
        self.production_data['recent_7day_abort_sum'] = self.production_data.groupby('pigfarm_dk')['abort_qty']\
                                                            .rolling(window=7, min_periods=7).sum()\
                                                            .reset_index(level=0, drop=True) # reset_index 去掉 groupby 带来的多级索引

        # 定义一个函数来计算流产率，处理分母为0的情况
        def calculate_rate(row):
            sum_recent_abort = row['recent_7day_abort_sum']
            current_preg_stock = row['preg_stock_qty']

            # 如果7天流产总和是NaN (因为窗口不足7天)，则流产率也是NaN
            if pd.isna(sum_recent_abort):
                return np.nan
            
            # 如果怀孕母猪存栏量是NaN，则流产率也是NaN
            if pd.isna(current_preg_stock):
                return np.nan
            
            numerator = sum_recent_abort
            denominator = sum_recent_abort + current_preg_stock
            
            if denominator == 0:
                return np.nan  # 或者 np.nan，根据业务需求决定如何处理分母为0的情况
            else:
                return numerator / denominator

        # 应用计算函数
        self.production_data['abortion_rate'] = self.production_data.apply(calculate_rate, axis=1)

    
    def calculate_abortion_feature(self):
        """
        计算流产率特征:
        1. abortion_feature_1_7: 流产率
        2. abortion_mean_recent_7d: 近7天流产率均值
        3. abortion_mean_recent_14d: 近14天流产率均值
        4. abortion_mean_recent_21d: 近21天流产率均值
        """
        self.production_data['abortion_feature_1_7'] = self.production_data['abortion_rate']
        self.production_data['abortion_mean_recent_7d'] = self.production_data.groupby('pigfarm_dk')['abortion_rate']\
            .rolling(window=7, min_periods=7).mean()\
            .reset_index(level=0, drop=True)
        self.production_data['abortion_mean_recent_14d'] = self.production_data.groupby('pigfarm_dk')['abortion_rate']\
            .rolling(window=14, min_periods=14).mean()\
            .reset_index(level=0, drop=True)
        self.production_data['abortion_mean_recent_21d'] = self.production_data.groupby('pigfarm_dk')['abortion_rate']\
            .rolling(window=21, min_periods=21).mean()\
            .reset_index(level=0, drop=True)
        
    
    def calculate_boar_feature(self):
        """
        计算种猪类型特征:
        1. boar_transin_times_30d: 30天内猪只转入次数 (boar_transin_qty不为0的天数)
        2. boar_transin_qty_30d: 30天内猪只转入总量
        3. boar_transin_ratio_30d_1: 转入猪只占比1 (基于当前存栏)
        4. boar_transin_ratio_30d_2: 转入猪只占比2 (基于30天量)
        """
        # 确保相关列是数值类型
        columns_to_convert = ['boar_transin_qty', 'basesow_qty', 'basempig_sqty', 
                              'reserve_sow_qty', 'reserve_mpig_qty']
        
        for col in columns_to_convert:
            if col in self.production_data.columns:
                self.production_data[col] = pd.to_numeric(self.production_data[col], errors='coerce').fillna(0)
            else:
                logger.warning(f"列 {col} 不存在，将创建并填充为0")
                self.production_data[col] = 0
        
        # 1. 计算30天内boar_transin_qty不为0的天数
        # 先创建一个标记，1表示当天有转入，0表示没有
        self.production_data['has_boar_transin'] = (self.production_data['boar_transin_qty'] > 0).astype(int)
        
        # 使用rolling计算30天内的转入次数
        self.production_data['boar_transin_times_30d'] = self.production_data.groupby('pigfarm_dk')['has_boar_transin']\
            .rolling(window=30, min_periods=1).sum()\
            .reset_index(level=0, drop=True)
        
        # 2. 计算30天内boar_transin_qty的和
        self.production_data['boar_transin_qty_30d'] = self.production_data.groupby('pigfarm_dk')['boar_transin_qty']\
            .rolling(window=30, min_periods=1).sum()\
            .reset_index(level=0, drop=True)
        
        # 3. 为第二个比率计算30天其他指标
        self.production_data['basesow_sqty_30d_ago'] = self.production_data.groupby('pigfarm_dk')['basesow_sqty'].shift(30)

        self.production_data['basempig_sqty_30d_ago'] = self.production_data.groupby('pigfarm_dk')['basempig_sqty'].shift(30)

        self.production_data['reserve_sow_sqty_30d_ago'] = self.production_data.groupby('pigfarm_dk')['reserve_sow_sqty'].shift(30)

        self.production_data['reserve_mpig_sqty_30d_ago'] = self.production_data.groupby('pigfarm_dk')['reserve_mpig_sqty'].shift(30)

        # 4. 计算boar_transin_ratio_30d_1 (基于当前存栏)
        def calculate_boar_ratio(row):
            numerator = row['boar_transin_qty_30d']
            denominator = (row['basesow_sqty'] + row['basempig_sqty'] + 
                          row['reserve_sow_sqty'] + row['reserve_mpig_sqty'] + 
                          row['boar_transin_qty_30d'])
            
            if denominator == 0:
                return 0  # 分母为0时返回0
            else:
                return numerator / denominator
        
        self.production_data['boar_transin_ratio_30d_1'] = self.production_data.apply(calculate_boar_ratio, axis=1)
        
        # 5. 计算boar_transin_ratio_30d_2 (基于30天前数量)
        def calculate_boar_ratio_30d(row):
            numerator = row['boar_transin_qty_30d']
            denominator = (row['basesow_sqty_30d_ago'] + row['basempig_sqty_30d_ago'] + 
                          row['basempig_sqty_30d_ago'] + row['reserve_sow_sqty_30d_ago'] + 
                          row['boar_transin_qty_30d'])
            
            if denominator == 0:
                return 0  # 分母为0时返回0
            else:
                return numerator / denominator
        
        self.production_data['boar_transin_ratio_30d_2'] = self.production_data.apply(calculate_boar_ratio_30d, axis=1)
        
        logger.info("种猪特征计算完成")


    def calculate_preg_stock_feature(self):
        """
        计算怀孕母猪特征:
        1. preg_stock_sqty_change_ratio_7d: 与7天前（T-6）相比的存栏量变化率
        2. preg_stock_sqty_change_ratio_15d: 与15天前（T-14）相比的存栏量变化率 
        3. preg_stock_sqty: preg_stock_qty对应数据
        """
        # 确保preg_stock_qty是数值类型
        self.production_data['preg_stock_qty'] = pd.to_numeric(self.production_data['preg_stock_qty'], errors='coerce').fillna(0)

        # 按猪场分组并计算6天和14天前的存栏量
        self.production_data['preg_stock_qty_6d_ago'] = self.production_data.groupby('pigfarm_dk')['preg_stock_qty'].shift(6)
        self.production_data['preg_stock_qty_14d_ago'] = self.production_data.groupby('pigfarm_dk')['preg_stock_qty'].shift(14)

        # 计算变化率
        def calculate_change_ratio(row, days):
            current = row['preg_stock_qty']
            previous = row[f'preg_stock_qty_{days}d_ago']
            
            # 处理缺失值
            if pd.isna(previous):
                return np.nan
                
            # 处理当前值为0的情况
            if current == 0:
                return np.nan
            
            # 计算变化率
            return (current - previous) / current
        
        # 应用计算函数
        self.production_data['preg_stock_sqty_change_ratio_7d'] = self.production_data.apply(
            lambda row: calculate_change_ratio(row, 6), axis=1)

        self.production_data['preg_stock_sqty_change_ratio_15d'] = self.production_data.apply(
            lambda row: calculate_change_ratio(row, 14), axis=1)
        
        # 计算后备母猪存栏量
        self.production_data['preg_stock_sqty'] = self.production_data['preg_stock_qty']

        logger.info("怀孕母猪特征计算完成")


    def calculate_reserve_sow_feature(self):
        """
        计算后备母猪特征:
        1. reserve_sow_sqty: 后备母猪存栏量
        2. reserve_sow_sqty_change_ratio_7d: 与7天前（T-6）相比的存栏量变化率
        3. reserve_sow_sqty_change_ratio_15d: 与15天前（T-14）相比的存栏量变化率
        """
        # 确保reserve_sow_qty是数值类型
        self.production_data['reserve_sow_sqty'] = pd.to_numeric(self.production_data['reserve_sow_sqty'], errors='coerce').fillna(0)
        
        # 按猪场分组并计算7天和15天前的存栏量
        self.production_data['reserve_sow_sqty_6d_ago'] = self.production_data.groupby('pigfarm_dk')['reserve_sow_sqty'].shift(6)
        self.production_data['reserve_sow_sqty_14d_ago'] = self.production_data.groupby('pigfarm_dk')['reserve_sow_sqty'].shift(14)
        
        # 计算变化率
        def calculate_change_ratio(row, days):
            current = row['reserve_sow_sqty']
            previous = row[f'reserve_sow_sqty_{days}d_ago']
            
            # 处理缺失值
            if pd.isna(previous):
                return np.nan
                
            # 处理当前值为0的情况
            if current == 0:
                return np.nan

            # 计算变化率
            return (current - previous) / current
        
        # 应用计算函数
        self.production_data['reserve_sow_sqty_change_ratio_7d'] = self.production_data.apply(
            lambda row: calculate_change_ratio(row, 6), axis=1)
        
        self.production_data['reserve_sow_sqty_change_ratio_15d'] = self.production_data.apply(
            lambda row: calculate_change_ratio(row, 14), axis=1)
        
        # 直接使用reserve_sow_sqty作为reserve_sow_sqty
        self.production_data['reserve_sow_sqty'] = self.production_data['reserve_sow_sqty']
        
        logger.info("后备母猪特征计算完成")


    def calculate_basesow_feature(self):
        """
        计算基础母猪特征:
        1. basesow_sqty: 基础母猪存栏量
        2. basesow_sqty_change_ratio_7d: 与7天前（T-6）相比的存栏量变化率
        3. basesow_sqty_change_ratio_15d: 与15天前（T-14）相比的存栏量变化率
        """
        # 确保basesow_qty是数值类型
        self.production_data['basesow_sqty'] = pd.to_numeric(self.production_data['basesow_sqty'], errors='coerce').fillna(0)

        # 按猪场分组并计算6天和14天前的存栏量
        self.production_data['basesow_sqty_6d_ago'] = self.production_data.groupby('pigfarm_dk')['basesow_sqty'].shift(6)
        self.production_data['basesow_sqty_14d_ago'] = self.production_data.groupby('pigfarm_dk')['basesow_sqty'].shift(14)
        
        # 计算变化率
        def calculate_change_ratio(row, days):
            current = row['basesow_sqty']
            previous = row[f'basesow_sqty_{days}d_ago']
            
            # 处理缺失值
            if pd.isna(previous):
                return np.nan
                
            # 处理当前值为0的情况
            if current == 0:
                return None
            
            # 计算变化率
            return (current - previous) / current
        
        # 应用计算函数
        self.production_data['basesow_sqty_change_ratio_7d'] = self.production_data.apply(
            lambda row: calculate_change_ratio(row, 6), axis=1)
        
        self.production_data['basesow_sqty_change_ratio_15d'] = self.production_data.apply(
            lambda row: calculate_change_ratio(row, 14), axis=1)
        
        # 直接使用basesow_qty作为basesow_sqty
        self.production_data['basesow_sqty'] = self.production_data['basesow_sqty']
        
        logger.info("基础母猪特征计算完成")


    def calculate_production_feature(self):
        if self.production_data is None:
            logger.error("无法计算生产特征，数据未加载")
            return None
        # 计算流产率
        self.calculate_abortion_rate()
        # 计算流产特征
        self.calculate_abortion_feature()
        # 计算种猪特征
        self.calculate_boar_feature()
        # 计算怀孕母猪特征
        self.calculate_preg_stock_feature()
        # 计算后备母猪特征
        self.calculate_reserve_sow_feature()
        # 计算基础母猪特征
        self.calculate_basesow_feature()
        # 选择需要的特征列
        feature_columns = ['stats_dt', 'pigfarm_dk'] + self.abortion_feature_columns + \
                            self.boar_feature_columns + self.preg_stock_feature_columns + \
                            self.reserve_sow_feature_columns + self.basesow_feature_columns
        production_feature = self.production_data[feature_columns].copy()
        # 确保数据按日期和猪场排序
        production_feature.sort_values(by=['stats_dt', 'pigfarm_dk'], inplace=True)
        production_feature.to_csv("production_feature.csv", index=False)
        assert False
        # 日期加1用于模拟当前没有数据
        production_feature['stats_dt'] = production_feature['stats_dt'] + pd.Timedelta(days=1)
        # 将特征合并到index_data上
        if self.index_data is not None:
            self.index_data = self.index_data.merge(production_feature, on=['stats_dt', 'pigfarm_dk'], how='left')
        
        return self.index_data

