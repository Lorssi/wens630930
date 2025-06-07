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


class DeathConfirmFeature(BaseFeatureDataSet):

    def __init__(self, running_dt_end: str, train_interval: int, file_type: str, **param):
        super().__init__(param)
        logger.info('-----Loading data-----')
        self.death_confirm_data = pd.read_csv(RawData.TMP_ORG_PRRS_OVERALL_ADOPT_CULL_DAY.value, encoding='utf-8-sig')

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
        death_confirm_data = self.death_confirm_data.copy()
        
        death_confirm_data['stats_dt'] = pd.to_datetime(death_confirm_data['stats_dt'])
        death_confirm_data = death_confirm_data[(death_confirm_data['stats_dt'] >= self.start_date) & (death_confirm_data['stats_dt'] <= self.end_date)]
        
        # 更新
        self.death_confirm_data = death_confirm_data

    def _init_entity_and_features(self):
        """初始化实体和特征定义"""
        # 设置实体列表
        self.entity = ['stats_dt', 'org_inv_dk']
        # 定义特征
        features_config = [
            # 月份特征
            ('death_confirm_2_week', FeatureType.Continuous, FeatureDtype.Int32, 'death_confirm_2_week'),
            # 季节特征
            ('death_confirm_5_week', FeatureType.Continuous, FeatureDtype.Int32, 'death_confirm_2_week'),
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

    def _get_death_confirm_feature(self):
        """获取组织特征"""
        death_confirm_data = self.death_confirm_data.copy()

        """
        计算动态两周死淘率和动态五周死淘率特征
        """
        if death_confirm_data.empty:
            logger.warning("死亡确认数据为空，无法计算特征")
            death_confirm_data['death_confirm_2_week'] = np.nan
            death_confirm_data['death_confirm_5_week'] = np.nan
        
        # 确保数据类型正确
        death_confirm_data['stats_dt'] = pd.to_datetime(death_confirm_data['stats_dt'])
        # death_confirm_data['stats_dt'] = death_confirm_data['stats_dt'] + pd.Timedelta(days=1)  # 确保日期是正确的
        
        death_confirm_data.rename(columns={'org_inv_dk': 'pigfarm_dk'}, inplace=True)

        
        # 预处理数据，只保留需要的列，提高处理效率
        death_data_processed = death_confirm_data[
            ['pigfarm_dk', 'stats_dt', 
             'ago_14_days_adopt_qty', 'ago_14_days_adopt_qty_sum', 'ago_14_days_cull_qty',
             'ago_35_days_adopt_qty', 'ago_35_days_adopt_qty_sum', 'ago_35_days_cull_qty']
        ].copy()
        
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
        
        self.data = result_data
       

    def _post_processing_data(self):
        if self.data.isnull().any().any():
            logger.info("Warning: Null in death_confirm_feature_data.csv")
        self.file_name = "death_confirm_feature_data." + self.file_type

        data = self.data.copy()
        
        data['stats_dt'] = data['stats_dt'] + pd.DateOffset(days=1)  # 确保日期是正确的
        self.data = data.copy()

    def build_dataset_all(self):
        logger.info("-----Preprocessing data----- ")
        self._preprocessing_data()
        logger.info("Calculating interval from last purchase...")
        self._get_death_confirm_feature()
        logger.info("-----Postprocessing data----- ")
        self._post_processing_data()
        # logger.info("-----Save as : {}".format("/".join([config.FEATURE_STORE_ROOT, self.file_name])))
        logger.info("-----Save as : {}".format(config.FeatureData.DEATH_CONFIRM_FEATURE_DATA.value))
        # self.dump_dataset("/".join([config.FEATURE_STORE_ROOT, self.file_name]))
        self.dump_dataset(config.FeatureData.DEATH_CONFIRM_FEATURE_DATA.value)
        logger.info("-----Dataset saved successfully-----")

if __name__ == "__main__":
    # Example usage
    running_dt_end = "2024-06-01"
    train_interval = 100
    file_type = "csv"



