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

from configs.base_config import RawData, FeatureData
from configs.pigfarm_risk_prediction_config import FEATURE_STORE_DIR
from tqdm import tqdm


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


class SurroundingPigfarmInfoFeature(BaseFeatureDataSet):

    def __init__(self, running_dt_end: str, train_interval: int, file_type: str, **param):
        super().__init__(param)
        logger.info('-----Loading data-----')
        self.abortion_data = pd.read_csv(FeatureData.PRODUCTION_FEATURE_DATA.value, encoding='utf-8-sig')
        self.org_data = pd.read_csv(RawData.DIM_ORG_INV.value, encoding='utf-8-sig')
        self.org_abortion_data = None

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
        abortion_data = self.abortion_data.copy()
        org_data = self.org_data.copy()
        org_data = org_data.rename(columns={'org_inv_dk': 'pigfarm_dk'})

        abortion_data['stats_dt'] = pd.to_datetime(abortion_data['stats_dt'])
        abortion_data = abortion_data[['stats_dt', 'pigfarm_dk', 'abortion_rate']]
        org_data = org_data[['pigfarm_dk', 'l3_org_inv_dk']]
        
        abortion_data = abortion_data.drop_duplicates(subset=['stats_dt', 'pigfarm_dk'], keep='first')

        org_abortion_data = pd.merge(
            abortion_data[['stats_dt', 'pigfarm_dk', 'abortion_rate']],
            org_data[['pigfarm_dk', 'l3_org_inv_dk']],
            on='pigfarm_dk',
            how='left'
        )
        # 更新
        self.org_abortion_data = org_abortion_data

    def _init_entity_and_features(self):
        """初始化实体和特征定义"""
        # 设置实体列表
        self.entity = ['stats_dt', 'pigfarm_dk']
        # 定义特征
        features_config = [
            # 组织特征
            ('l3_abortion_mean', FeatureType.Continuous, FeatureDtype.Float32, 'sorrounding'),
            # 流产率特征
            ('l3_abortion_mean_7d', FeatureType.Continuous, FeatureDtype.Float32, 'sorrounding'),
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

    def _get_sorrounding_feature(self):
        # 准备数据
        org_abortion_data = self.org_abortion_data.copy()
        # 确保stats_dt是datetime类型
        org_abortion_data['stats_dt'] = pd.to_datetime(org_abortion_data['stats_dt'])
        
        # 1. 计算同三级公司同天的平均流产率
        l3_same_day_avg = org_abortion_data.groupby(['l3_org_inv_dk', 'stats_dt'])['abortion_rate'].mean().reset_index()
        l3_same_day_avg.rename(columns={'abortion_rate': 'l3_abortion_mean'}, inplace=True)
        
        # 2. 计算同三级公司近7天的平均流产率（使用高效的方式）
        # 按三级公司和日期排序
        org_abortion_data = org_abortion_data.sort_values(['l3_org_inv_dk', 'stats_dt'])
        
        # 按三级公司分组计算7天窗口均值
        rolling_results = []
        
        for l3_id, group_data in tqdm(org_abortion_data.groupby('l3_org_inv_dk'), desc="计算滚动平均"):
            # 按日期分组，计算每天的平均值（避免同一天多个猪场重复计算）
            daily_avg = group_data.groupby('stats_dt')['abortion_rate'].mean().reset_index()
            daily_avg = daily_avg.sort_values('stats_dt').set_index('stats_dt')
            
            # 使用rolling函数计算7天窗口的平均值
            daily_avg['l3_abortion_mean_7d'] = daily_avg['abortion_rate'].rolling(window='7D', min_periods=7).mean()
            daily_avg['l3_abortion_mean_15d'] = daily_avg['abortion_rate'].rolling(window='15D', min_periods=15).mean()
            daily_avg['l3_abortion_mean_30d'] = daily_avg['abortion_rate'].rolling(window='30D', min_periods=30).mean()
            
            daily_avg = daily_avg.reset_index()
            daily_avg['l3_org_inv_dk'] = l3_id
            rolling_results.append(daily_avg[['l3_org_inv_dk', 'stats_dt', 'l3_abortion_mean_7d','l3_abortion_mean_15d', 'l3_abortion_mean_30d']])
        
        # 合并所有结果
        rolling_avg_df = pd.concat(rolling_results, ignore_index=True)
        
        index_data = self.org_abortion_data.copy()
        # 合并结果到原数据
        index_data = index_data.merge(
            l3_same_day_avg,
            on=['l3_org_inv_dk', 'stats_dt'],
            how='left'
        )
        
        index_data = index_data.merge(
            rolling_avg_df[['l3_org_inv_dk', 'stats_dt', 'l3_abortion_mean_7d','l3_abortion_mean_15d', 'l3_abortion_mean_30d']],
            on=['l3_org_inv_dk', 'stats_dt'],
            how='left'
        )

        self.data = index_data.copy()


    def _post_processing_data(self):
        data = self.data.copy()

        if data.isnull().any().any():
            logger.info("Warning: Null in production_feature_data.csv")
        self.file_name = "production_feature_data." + self.file_type

        keep_cols = ['stats_dt', 'pigfarm_dk', 'l3_abortion_mean', 'l3_abortion_mean_7d', 'l3_abortion_mean_15d', 'l3_abortion_mean_30d']
        data = data[keep_cols]
        self.data = data.copy()

    def build_dataset_all(self):
        logger.info("-----Preprocessing data----- ")
        self._preprocessing_data()
        logger.info("Calculating interval from last purchase...")
        logger.info("-----get sorrounding rate-----")
        self._get_sorrounding_feature()
        logger.info("-----Postprocessing data----- ")
        self._post_processing_data()
        # logger.info("-----Save as : {}".format("/".join([config.FEATURE_STORE_ROOT, self.file_name])))
        logger.info("-----Save as : {}".format(config.FeatureData.SORROUNDING_FEATURE_DATA.value))
        # self.dump_dataset("/".join([config.FEATURE_STORE_ROOT, self.file_name]))
        self.dump_dataset(config.FeatureData.SORROUNDING_FEATURE_DATA.value)
        logger.info("-----Dataset saved successfully-----")

if __name__ == "__main__":
    # Example usage
    running_dt_end = "2024-06-01"
    train_interval = 100
    file_type = "csv"

    dataset = ProductionFeature(running_dt_end, train_interval, file_type)
    dataset.build_dataset_all()
    logger.info("Production feature dataset built successfully.")



