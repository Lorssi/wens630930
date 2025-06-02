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

    def _post_processing_data(self):
        if self.data.isnull().any().any():
            logger.info("Warning: Null in production_feature_data.csv")
        self.file_name = "production_feature_data." + self.file_type

    def build_dataset_all(self):
        logger.info("-----Preprocessing data----- ")
        self._preprocessing_data()
        logger.info("Calculating interval from last purchase...")
        self._get_abortion_rate()
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



