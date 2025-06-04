import os
import sys
import logging

import pandas as pd
import numpy as np

from dataset.base_dataset import BaseDataSet
from dataset.base_feature_dataset import BaseFeatureDataSet
import configs.base_config as config
from datetime import timedelta, datetime
from transform.features import FeatureType, FeatureDtype, Feature

from configs.base_config import RawData


# 显示所有列
pd.set_option('display.max_columns', None)
# 显示所有行
pd.set_option('display.max_rows', None)
# 设置value的显示长度为100，默认为50
pd.set_option('max_colwidth', 100)

base_dir = os.path.split(os.path.realpath(__file__))[0]
program = os.path.basename(sys.argv[0])
logger = logging.getLogger(program)
logging.basicConfig(format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s: %(message)s',
                    datefmt='%a, %d %b %Y %H:%M:%S')
logging.root.setLevel(level=logging.INFO)


class ProductionFeature(BaseFeatureDataSet):

    def __init__(self, running_dt_end: str, train_interval: int, file_type: str, **param):
        super().__init__(param)
        logger.info('-----Loading data-----')
        self.production_data = pd.read_csv(RawData.ADS_PIG_ORG_TOTAL_TO_ML_TRAINING_DAY.value, encoding='utf-8-sig')

        self.running_dt_end = running_dt_end
        self.train_interval = train_interval
        self.file_type = file_type

        self.end_date = (datetime.strptime(self.running_dt_end, "%Y-%m-%d") - timedelta(days=1)).strftime("%Y-%m-%d")
        self.start_date = (datetime.strptime(self.end_date, "%Y-%m-%d") - timedelta(days=self.train_interval) - timedelta(days=40)).strftime("%Y-%m-%d")
        logger.info('-----start_date: {}'.format(self.start_date))
        logger.info('-----end_date: {}'.format(self.end_date))

        self.data = pd.DataFrame()
        self.file_name = None

        self._init_entity_and_features()

    def _preprocessing_data(self):
        production_data = self.production_data.copy()

        production_data['stats_dt'] = pd.to_datetime(production_data['stats_dt'])
        production_data = production_data[
            (production_data['stats_dt'] >= self.start_date) &
            (production_data['stats_dt'] <= self.end_date)]
        
        production_data = production_data.drop_duplicates(subset=['stats_dt', 'pigfarm_dk'], keep='first')
        production_data.sort_values(['stats_dt', 'pigfarm_dk'], inplace=True)
        # 更新
        self.production_data = production_data

    def _init_entity_and_features(self):
        """初始化实体和特征定义"""
        # 设置实体列表
        self.entity = ['stats_dt', 'pigfarm_dk']
        # 定义特征
        features_config = [
            # 组织特征
            ('pigfarm_dk', FeatureType.Categorical, FeatureDtype.String, 'organization'),
            # 流产率特征
            ('abortion_rate', FeatureType.Continuous, FeatureDtype.Float32, 'abortion'),
        ]

        # 创建并添加特征到 features 对象
        for name, feature_type, dtype, domain in features_config:
            feature = Feature(
                name=name,
                domain=domain,
                feature_type=feature_type,
                dtype=dtype
            )
            self.features.add(feature)

        logger.info(f"初始化完成 - 实体数量: {len(self.entity)}, 特征数量: {len(self.features)}")

    def _get_abortion_rate(self):
        """
        计算流产率。
        流产率 = sum(近7天流产数量) / (sum(近7天流产数量) + 当天怀孕母猪存栏量)
        """
        abort_qty_column = 'abort_qty'
        preg_stock_qty_column = 'preg_stock_qty'
        id_column = 'pigfarm_dk'
        date_column = 'stats_dt'

        feature_name = 'abortion_rate'

        production_data = self.production_data.copy()

        # 确保流产数量和怀孕母猪存栏量是数值类型，并将NaN填充为0，因为它们参与计算
        production_data[abort_qty_column] = pd.to_numeric(production_data[abort_qty_column], errors='coerce').fillna(0)
        production_data[preg_stock_qty_column] = pd.to_numeric(production_data[preg_stock_qty_column], errors='coerce').fillna(0)
        
        # 使用 groupby 和 rolling window 计算每个猪场每个日期的近7天流产总数
        # 'closed="left"' 通常用于rolling sum，但这里我们需要包含当天，所以默认'right'就可以
        # min_periods=1 表示即使不足7天，也会计算已有的天数和
        production_data['recent_7day_abort_sum'] = production_data.groupby(id_column)[abort_qty_column]\
                                                            .rolling(window=7, min_periods=7).sum()\
                                                            .reset_index(level=0, drop=True) # reset_index 去掉 groupby 带来的多级索引

        # 定义一个函数来计算流产率，处理分母为0的情况
        def calculate_rate(row):
            sum_recent_abort = row['recent_7day_abort_sum']
            current_preg_stock = row[preg_stock_qty_column]

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
        production_data[feature_name] = production_data.apply(calculate_rate, axis=1)

        self.data = production_data.copy()

    def _get_abortion_mean_feature(self):
        """
        计算流产率特征:
        1. abortion_feature_1_7: 流产率
        2. abortion_mean_recent_7d: 近7天流产率均值
        3. abortion_mean_recent_14d: 近14天流产率均值
        4. abortion_mean_recent_21d: 近21天流产率均值
        """
        data = self.data.copy()

        data['abortion_mean_recent_7d'] = data.groupby('pigfarm_dk')['abortion_rate']\
            .rolling(window=7, min_periods=7).mean()\
            .reset_index(level=0, drop=True)
        data['abortion_mean_recent_14d'] = data.groupby('pigfarm_dk')['abortion_rate']\
            .rolling(window=14, min_periods=14).mean()\
            .reset_index(level=0, drop=True)
        data['abortion_mean_recent_21d'] = data.groupby('pigfarm_dk')['abortion_rate']\
            .rolling(window=21, min_periods=21).mean()\
            .reset_index(level=0, drop=True)

        self.data = data.copy()

    def _get_abortion_mean_feature(self):
        """
        计算流产率特征:
        1. abortion_feature_1_7: 流产率
        2. abortion_mean_recent_7d: 近7天流产率均值
        3. abortion_mean_recent_14d: 近14天流产率均值
        4. abortion_mean_recent_21d: 近21天流产率均值
        """
        data = self.data.copy()

        data['abortion_mean_recent_7d'] = data.groupby('pigfarm_dk')['abortion_rate']\
            .rolling(window=7, min_periods=7).mean()\
            .reset_index(level=0, drop=True)
        data['abortion_mean_recent_14d'] = data.groupby('pigfarm_dk')['abortion_rate']\
            .rolling(window=14, min_periods=14).mean()\
            .reset_index(level=0, drop=True)
        data['abortion_mean_recent_21d'] = data.groupby('pigfarm_dk')['abortion_rate']\
            .rolling(window=21, min_periods=21).mean()\
            .reset_index(level=0, drop=True)

        self.data = data.copy()

    def _get_boar_feature(self):
        """
        计算种猪类型特征:
        1. boar_transin_times_30d: 30天内猪只转入次数 (boar_transin_qty不为0的天数)
        2. boar_transin_qty_30d: 30天内猪只转入总量
        3. boar_transin_ratio_30d_1: 转入猪只占比1 (基于当前存栏)
        4. boar_transin_ratio_30d_2: 转入猪只占比2 (基于30天量)
        """
        data = self.data.copy()
        # 确保相关列是数值类型
        columns_to_convert = ['boar_transin_qty', 'basesow_sqty', 'basempig_sqty', 
                              'reserve_sow_sqty', 'reserve_mpig_sqty']
        
        for col in columns_to_convert:
            if col in data.columns:
                data[col] = pd.to_numeric(data[col], errors='coerce').fillna(0)
            else:
                logger.warning(f"列 {col} 不存在，将创建并填充为0")
                data[col] = np.nan
        
        # 1. 计算30天内boar_transin_qty不为0的天数
        # 先创建一个标记，1表示当天有转入，0表示没有
        data['has_boar_transin'] = (data['boar_transin_qty'] > 0).astype(int)
        
        # 使用rolling计算30天内的转入次数
        data['boar_transin_times_30d'] = data.groupby('pigfarm_dk')['has_boar_transin']\
            .rolling(window=30, min_periods=1).sum()\
            .reset_index(level=0, drop=True)
        
        # 2. 计算30天内boar_transin_qty的和
        data['boar_transin_qty_30d'] = data.groupby('pigfarm_dk')['boar_transin_qty']\
            .rolling(window=30, min_periods=1).sum()\
            .reset_index(level=0, drop=True)
        
        # 3. 为第二个比率计算30天其他指标
        data['basesow_sqty_30d_ago'] = data.groupby('pigfarm_dk')['basesow_sqty'].shift(30)

        data['basempig_sqty_30d_ago'] = data.groupby('pigfarm_dk')['basempig_sqty'].shift(30)

        data['reserve_sow_sqty_30d_ago'] = data.groupby('pigfarm_dk')['reserve_sow_sqty'].shift(30)

        data['reserve_mpig_sqty_30d_ago'] = data.groupby('pigfarm_dk')['reserve_mpig_sqty'].shift(30)

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
        
        data['boar_transin_ratio_30d_1'] = data.apply(calculate_boar_ratio, axis=1)
        
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
        
        data['boar_transin_ratio_30d_2'] = data.apply(calculate_boar_ratio_30d, axis=1)
        
        self.data = data.copy()
        logger.info("种猪特征计算完成")

    def _get_preg_stock_feature(self):
        """
        计算怀孕母猪特征:
        1. preg_stock_sqty_change_ratio_7d: 与7天前（T-6）相比的存栏量变化率
        2. preg_stock_sqty_change_ratio_15d: 与15天前（T-14）相比的存栏量变化率 
        3. preg_stock_sqty: preg_stock_qty对应数据
        """
        data = self.data.copy()
        # 确保preg_stock_qty是数值类型
        data['preg_stock_qty'] = pd.to_numeric(data['preg_stock_qty'], errors='coerce').fillna(0)

        # 按猪场分组并计算6天和14天前的存栏量
        data['preg_stock_qty_6d_ago'] = data.groupby('pigfarm_dk')['preg_stock_qty'].shift(6)
        data['preg_stock_qty_14d_ago'] = data.groupby('pigfarm_dk')['preg_stock_qty'].shift(14)

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
        data['preg_stock_sqty_change_ratio_7d'] = data.apply(
            lambda row: calculate_change_ratio(row, 6), axis=1)

        data['preg_stock_sqty_change_ratio_15d'] = data.apply(
            lambda row: calculate_change_ratio(row, 14), axis=1)
        
        # 计算后备母猪存栏量
        data['preg_stock_sqty'] = data['preg_stock_qty']

        self.data = data.copy()
        logger.info("怀孕母猪特征计算完成")

    def _get_reserve_sow_feature(self):
        """
        计算后备母猪特征:
        1. reserve_sow_sqty: 后备母猪存栏量
        2. reserve_sow_sqty_change_ratio_7d: 与7天前（T-6）相比的存栏量变化率
        3. reserve_sow_sqty_change_ratio_15d: 与15天前（T-14）相比的存栏量变化率
        """
        data = self.data.copy()
        # 确保reserve_sow_qty是数值类型
        data['reserve_sow_sqty'] = pd.to_numeric(data['reserve_sow_sqty'], errors='coerce').fillna(0)
        
        # 按猪场分组并计算7天和15天前的存栏量
        data['reserve_sow_sqty_6d_ago'] = data.groupby('pigfarm_dk')['reserve_sow_sqty'].shift(6)
        data['reserve_sow_sqty_14d_ago'] = data.groupby('pigfarm_dk')['reserve_sow_sqty'].shift(14)
        
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
        data['reserve_sow_sqty_change_ratio_7d'] = data.apply(
            lambda row: calculate_change_ratio(row, 6), axis=1)
        
        data['reserve_sow_sqty_change_ratio_15d'] = data.apply(
            lambda row: calculate_change_ratio(row, 14), axis=1)
        
        # 直接使用reserve_sow_sqty作为reserve_sow_sqty
        data['reserve_sow_sqty'] = data['reserve_sow_sqty']
        
        self.data = data.copy()
        logger.info("后备母猪特征计算完成")

    def _get_basesow_feature(self):
        """
        计算基础母猪特征:
        1. basesow_sqty: 基础母猪存栏量
        2. basesow_sqty_change_ratio_7d: 与7天前（T-6）相比的存栏量变化率
        3. basesow_sqty_change_ratio_15d: 与15天前（T-14）相比的存栏量变化率
        """
        data = self.data.copy()
        # 确保basesow_qty是数值类型
        data['basesow_sqty'] = pd.to_numeric(data['basesow_sqty'], errors='coerce').fillna(0)

        # 按猪场分组并计算6天和14天前的存栏量
        data['basesow_sqty_6d_ago'] = data.groupby('pigfarm_dk')['basesow_sqty'].shift(6)
        data['basesow_sqty_14d_ago'] = data.groupby('pigfarm_dk')['basesow_sqty'].shift(14)
        
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
        data['basesow_sqty_change_ratio_7d'] = data.apply(
            lambda row: calculate_change_ratio(row, 6), axis=1)
        
        data['basesow_sqty_change_ratio_15d'] = data.apply(
            lambda row: calculate_change_ratio(row, 14), axis=1)
        
        # 直接使用basesow_qty作为basesow_sqty
        data['basesow_sqty'] = data['basesow_sqty']
        
        self.data = data.copy()
        logger.info("基础母猪特征计算完成")

    def _post_processing_data(self):
        data = self.data.copy()

        if data.isnull().any().any():
            logger.info("Warning: Null in production_feature_data.csv")
        self.file_name = "production_feature_data." + self.file_type

        production_feature_list = ['stats_dt', 'pigfarm_dk', 'abortion_rate','abortion_mean_recent_7d',
                                   'abortion_mean_recent_14d', 'abortion_mean_recent_21d',
                                   'boar_transin_times_30d', 'boar_transin_qty_30d',
                                   'boar_transin_ratio_30d_1', 'boar_transin_ratio_30d_2',
                                   'preg_stock_sqty_change_ratio_7d', 'preg_stock_sqty_change_ratio_15d','preg_stock_sqty',
                                   'reserve_sow_sqty_change_ratio_7d', 'reserve_sow_sqty_change_ratio_15d','reserve_sow_sqty',
                                   'basesow_sqty_change_ratio_7d', 'basesow_sqty_change_ratio_15d','basesow_sqty']
        data = data[production_feature_list]
        self.data = data.copy()

    def build_dataset_all(self):
        logger.info("-----Preprocessing data----- ")
        self._preprocessing_data()
        logger.info("Calculating interval from last purchase...")
        logger.info("-----get abortion rate-----")
        self._get_abortion_rate()
        logger.info("-----get abortion mean feature-----")
        self._get_abortion_mean_feature()
        logger.info("-----get boar feature-----")
        self._get_boar_feature()
        logger.info("-----get preg stock feature-----")
        self._get_preg_stock_feature()
        logger.info("-----get reserve sow feature-----")
        self._get_reserve_sow_feature()
        logger.info("-----get basesow feature-----")
        self._get_basesow_feature()
        logger.info("-----Postprocessing data----- ")
        self._post_processing_data()
        # logger.info("-----Save as : {}".format("/".join([config.FEATURE_STORE_ROOT, self.file_name])))
        logger.info("-----Save as : {}".format(config.FeatureData.PRODUCTION_FEATURE_DATA.value))
        # self.dump_dataset("/".join([config.FEATURE_STORE_ROOT, self.file_name]))
        self.dump_dataset(config.FeatureData.PRODUCTION_FEATURE_DATA.value)
        logger.info("-----Dataset saved successfully-----")

if __name__ == "__main__":
    # Example usage
    running_dt_end = "2024-06-01"
    train_interval = 100
    file_type = "csv"

    dataset = ProductionFeature(running_dt_end, train_interval, file_type)
    dataset.build_dataset_all()
    logger.info("Production feature dataset built successfully.")



