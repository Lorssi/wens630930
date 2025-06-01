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


class DateFeature(BaseFeatureDataSet):

    def __init__(self, running_dt_end: str, train_interval: int, file_type: str, **param):
        super().__init__(param)
        logger.info('-----Loading data-----')
        self.production_data = pd.read_csv(RawData.ADS_PIG_ORG_TOTAL_TO_ML_TRAINING_DAY.value, encoding='utf-8-sig')

        self.running_dt_end = running_dt_end
        self.train_interval = train_interval
        self.file_type = file_type

        self.end_date = (datetime.strptime(self.running_dt_end, "%Y-%m-%d") - timedelta(days=1)).strftime("%Y-%m-%d")
        self.start_date = (datetime.strptime(self.end_date, "%Y-%m-%d") - timedelta(days=self.train_interval) - timedelta(days=10)).strftime("%Y-%m-%d")
        logger.info('-----start_date: {}'.format(self.start_date))
        logger.info('-----end_date: {}'.format(self.end_date))

        self.data = pd.DataFrame()
        self.file_name = None

        self._init_entity_and_features()

    def _preprocessing_data(self):
        production_data = self.production_data[self.entity].copy()
        
        production_data['stats_dt'] = pd.to_datetime(production_data['stats_dt'])
        production_data = production_data[(production_data['stats_dt'] >= self.start_date) & (production_data['stats_dt'] <= self.end_date)]
        
        # 更新
        self.production_data = production_data

    def _init_entity_and_features(self):
        """初始化实体和特征定义"""
        # 设置实体列表
        self.entity = ['stats_dt', 'pigfarm_dk']
        # 定义特征
        features_config = [
            # 月份特征
            ('month', FeatureType.Categorical, FeatureDtype.Int32, 'date'),
            # 季节特征
            ('season', FeatureType.Categorical, FeatureDtype.Int32, 'season'),
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

    def _get_month_feature(self):
        """获取组织特征"""
        production_data = self.production_data.copy()
        production_data['month'] = production_data['stats_dt'].dt.month.astype(np.int32)
        
        self.data = pd.concat([self.data, production_data[self.entity + ['month']]], axis=1)

    def _get_season_feature(self):
        """获取季节特征"""
        data = self.data.copy()
        
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
        data['season'] = data['month'].map(season_map).astype(int)
        
        self.data = pd.concat([self.data, data[['season']]], axis=1)

    def _post_processing_data(self):
        if self.data.isnull().any().any():
            logger.info("Warning: Null in date_feature_data.csv")
        self.file_name = "date_feature_data." + self.file_type

    def build_dataset_all(self):
        logger.info("-----Preprocessing data----- ")
        self._preprocessing_data()
        logger.info("Calculating interval from last purchase...")
        self._get_month_feature()
        self._get_season_feature()
        logger.info("-----Postprocessing data----- ")
        self._post_processing_data()
        # logger.info("-----Save as : {}".format("/".join([config.FEATURE_STORE_ROOT, self.file_name])))
        logger.info("-----Save as : {}".format(config.FeatureData.DATE_FEATURE_DATA.value))
        # self.dump_dataset("/".join([config.FEATURE_STORE_ROOT, self.file_name]))
        self.dump_dataset(config.FeatureData.DATE_FEATURE_DATA.value)
        logger.info("-----Dataset saved successfully-----")

if __name__ == "__main__":
    # Example usage
    running_dt_end = "2024-06-01"
    train_interval = 100
    file_type = "csv"

    dataset = DateFeature(running_dt_end, train_interval, file_type)
    dataset.build_dataset_all()
    logger.info("Production feature dataset built successfully.")



