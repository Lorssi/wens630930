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
from tqdm import tqdm
import numpy as np


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


class IntroFeature(BaseFeatureDataSet):

    def __init__(self, running_dt_end: str, train_interval: int, file_type: str, **param):
        super().__init__(param)
        logger.info('-----Loading data-----')
        self.intro_data = pd.read_csv(RawData.W01_AST_BOAR.value, encoding='utf-8-sig')
        self.tame_data = pd.read_csv(RawData.TMP_ADS_PIG_ISOLATION_TAME_RISK_L1_N2.value, encoding='utf-8-sig', low_memory=False)
        self.index_data = pd.read_csv(RawData.ADS_PIG_ORG_TOTAL_TO_ML_TRAINING_DAY.value, encoding='utf-8-sig')
        self.intro_batch_data = pd.read_csv(RawData.ADS_PIG_ISOLATION_TAME_PROLINE_RISK.value, encoding='utf-8-sig', low_memory=False)

        self.running_dt_end = running_dt_end
        self.train_interval = train_interval
        self.file_type = file_type

        self.end_date = (datetime.strptime(self.running_dt_end, "%Y-%m-%d") - timedelta(days=1)).strftime("%Y-%m-%d")
        self.start_date = (datetime.strptime(self.end_date, "%Y-%m-%d") - timedelta(days=self.train_interval) - timedelta(days=92)).strftime("%Y-%m-%d")
        logger.info('-----start_date: {}'.format(self.start_date))
        logger.info('-----end_date: {}'.format(self.end_date))

        self.data = pd.DataFrame()
        self.file_name = None

        self.feature_columns = ['intro_source_num_90d', 'intro_source_is_single', 'intro_times_30d', 'intro_times_90d', 'intro_days_30d', 'intro_days_90d', 'intro_batch_num_30d', 'intro_batch_num_90d']

        self._init_entity_and_features()

    def _preprocessing_data(self):
        intro_data = self.intro_data.copy()
        tame_data = self.tame_data.copy()
        index_data = self.index_data.copy()
        intro_batch_data = self.intro_batch_data.copy()

        tame_data.rename(columns={'tmp_ads_pig_isolation_tame_risk_l1_n2.org_inv_dk': 'org_inv_dk', 'tmp_ads_pig_isolation_tame_risk_l1_n2.bill_dt': 'bill_dt'}, inplace=True)
        index_data = index_data[['stats_dt', 'pigfarm_dk']]
        intro_data['intro_dt'] = pd.to_datetime(intro_data['intro_dt'])
        tame_data['bill_dt'] = pd.to_datetime(tame_data['bill_dt'])
        index_data['stats_dt'] = pd.to_datetime(index_data['stats_dt'])
        intro_batch_data['allot_dt'] = pd.to_datetime(intro_batch_data['allot_dt'])

        # 过滤日期范围，减少数据大小
        intro_data = intro_data[(intro_data['intro_dt'] >= self.start_date) & (intro_data['intro_dt'] <= self.end_date)]
        tame_data = tame_data[(tame_data['bill_dt'] >= self.start_date) & (tame_data['bill_dt'] <= self.end_date)]
        index_data = index_data[(index_data['stats_dt'] >= self.start_date) & (index_data['stats_dt'] <= self.end_date)]
        intro_batch_data = intro_batch_data[(intro_batch_data['allot_dt'] >= self.start_date) & (intro_batch_data['allot_dt'] <= self.end_date)]

        # 把来源猪场的空值由供应商名字填充
        intro_data['cffromhogp_nm'] = intro_data['cffromhogp_nm'].fillna(intro_data['vendor_nm'])
        # 排除 boar_src_type 为 '选留' 的记录
        intro_data = intro_data[intro_data['boar_src_type'] != '选留']

        # 排序，用于加快计算速度
        intro_data.sort_values(['intro_dt','org_inv_dk'], inplace=True)
        # 排序，用于加快计算速度
        tame_data.sort_values(['bill_dt', 'org_inv_dk'], inplace=True)
        index_data.sort_values(['stats_dt', 'pigfarm_dk'], inplace=True)
        intro_batch_data.sort_values(['allot_dt', 'prorg_inv_dk'], inplace=True)


        
        # 更新
        self.intro_data = intro_data
        self.tame_data = tame_data
        self.index_data = index_data
        self.intro_batch_data = intro_batch_data

    def _init_entity_and_features(self):
        """初始化实体和特征定义"""
        # 设置实体列表
        self.entity = ['org_inv_dk']
        # 定义特征
        features_config = [
            # 引种特征
            ('intro_source_is_single', FeatureType.Categorical, FeatureDtype.Int32, 'intro'),
            ('intro_source_num_90d', FeatureType.Continuous, FeatureDtype.Int32, 'intro'),
            ('intro_times_30d', FeatureType.Continuous, FeatureDtype.Int32, 'intro'),
            ('intro_times_90d', FeatureType.Continuous, FeatureDtype.Int32, 'intro'),
            ('intro_batch_num_30d', FeatureType.Continuous, FeatureDtype.Int32, 'intro'),
            ('intro_batch_num_90d', FeatureType.Continuous, FeatureDtype.Int32, 'intro'),
            # 入群特征
            ('boar_tame_num_30d', FeatureType.Categorical, FeatureDtype.String, 'tame'),
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

    def _get_intro_feature(self):
        """
        计算引种特征 - 高效优化版，先预处理数据再计算特征
        """
        intro_data = self.intro_data.copy()
        index_data = self.index_data.copy()
        intro_batch_data = self.intro_batch_data.copy()
                
        # 获取所有需要计算特征的日期（T+1模式）
        all_stats_dates = sorted(index_data['stats_dt'].unique())
        # 获取需要计算的日期，扩大范围到前后5天
        min_date = min(all_stats_dates)
        max_date = max(all_stats_dates)
        all_calc_dates = pd.date_range(start=min_date, end=max_date)
        
        # 创建一个空DataFrame用于存储所有特征
        farm_ids = index_data['pigfarm_dk'].unique()
        
        # 初始化一个字典存储所有特征
        farm_features = {}
        
        # 为每个猪场预计算特征
        for farm_id in tqdm(farm_ids, desc="计算猪场特征"):
            # 筛选该猪场的所有引种记录
            farm_intro = intro_data[intro_data['org_inv_dk'] == farm_id]
            # 筛选该猪场的所有引种批次记录
            farm_batch_intro = intro_batch_data[intro_batch_data['prorg_inv_dk'] == farm_id]

            has_intro = not farm_intro.empty
            has_batch_intro = not farm_batch_intro.empty

            if not has_intro and not has_batch_intro:
                continue
        
            # 为这个猪场创建一个日期字典
            farm_dict = {}
                
            # 对每个日期计算特征 (只计算index_data中的日期)
            for date in all_calc_dates:
                feature_dict = {
                    'intro_source_num_90d': np.nan,
                    'intro_source_is_single': np.nan, 
                    'intro_source_num_90d': np.nan,
                    'intro_times_90d': np.nan,
                    'intro_times_30d': np.nan,
                    'intro_days_30d': np.nan,
                    'intro_days_90d': np.nan,
                    'intro_batch_num_30d': np.nan,
                    'intro_batch_num_90d': np.nan,
                }
                
                # 计算引种特征
                if has_intro:
                    # 创建90天和30天的过滤条件
                    mask_90d = (farm_intro['intro_dt'] <= date) & (farm_intro['intro_dt'] >= date - pd.Timedelta(days=90))
                    mask_30d = (farm_intro['intro_dt'] <= date) & (farm_intro['intro_dt'] >= date - pd.Timedelta(days=30))

                    # 利用已有的掩码，避免重复过滤数据
                    intro_days_30d = farm_intro.loc[mask_30d, 'intro_dt'].nunique()
                    intro_days_90d = farm_intro.loc[mask_90d, 'intro_dt'].nunique()
                    
                    # 直接计算特征值，无需创建中间DataFrame
                    if mask_90d.any():  # 如果有任何符合条件的行
                        # 计算90天内的唯一来源数
                        unique_sources = farm_intro.loc[mask_90d, 'cffromhogp_nm'].nunique()
                        
                        # 更新特征值
                        feature_dict.update({
                            'intro_source_num_90d': unique_sources,
                            'intro_source_is_single': 1 if unique_sources <= 1 else 0,
                            'intro_times_90d': mask_90d.sum(),  # 直接对掩码求和获取行数
                            'intro_times_30d': mask_30d.sum(),   # 直接对掩码求和获取行数
                            'intro_days_30d': intro_days_30d,
                            'intro_days_90d': intro_days_90d
                        })
                
                # 计算批次引种特征
                if has_batch_intro:
                    # 创建90天和30天的过滤条件
                    batch_mask_90d = (farm_batch_intro['allot_dt'] <= date) & (farm_batch_intro['allot_dt'] >= date - pd.Timedelta(days=90))
                    batch_mask_30d = (farm_batch_intro['allot_dt'] <= date) & (farm_batch_intro['allot_dt'] >= date - pd.Timedelta(days=30))

                    # 计算批次数量
                    if batch_mask_90d.any():
                        # 获取90天内符合条件的数据
                        batch_data_90d = farm_batch_intro.loc[batch_mask_90d, ['allot_dt', 'prorg_inv_dk', 'source_org_dk', 'source_kind', 'rearer_pop_dk']]
                        # 去重计算90天内批次数
                        intro_batch_num_90d = batch_data_90d.drop_duplicates().shape[0]
                        
                        # 获取30天内符合条件的数据
                        batch_data_30d = farm_batch_intro.loc[batch_mask_30d, ['allot_dt', 'prorg_inv_dk', 'source_org_dk', 'source_kind', 'rearer_pop_dk']]
                        # 去重计算30天内批次数
                        intro_batch_num_30d = batch_data_30d.drop_duplicates().shape[0]
                        
                        # 更新特征值
                        feature_dict.update({
                            'intro_batch_num_90d': intro_batch_num_90d,
                            'intro_batch_num_30d': intro_batch_num_30d
                        })
                    
                farm_dict[date] = feature_dict
                        
            # 存储这个猪场的所有特征
            farm_features[farm_id] = farm_dict
        
        logger.info("将预计算特征合并到结果中...")
        
        # 将farm_features转换为DataFrame
        if farm_features:
            # 使用列表推导式扁平化嵌套字典
            flattened_features = [
                {"pigfarm_dk": farm_id, "stats_dt": date, **feature_values}
                for farm_id, date_dict in farm_features.items()
                for date, feature_values in date_dict.items()
            ]

            # 转换为DataFrame
            features_df = pd.DataFrame(flattened_features)
            features_df = features_df[['stats_dt', 'pigfarm_dk'] + self.feature_columns]
            self.data = features_df


    def _post_processing_data(self):
        data = self.data.copy()
        if data.isnull().any().any():
            logger.info("Warning: Null in org_feature_data.csv")
        self.file_name = "intro_feature_data." + self.file_type

    def build_dataset_all(self):
        logger.info("-----Preprocessing data----- ")
        self._preprocessing_data()
        logger.info("Calculating interval from last purchase...")
        self._get_intro_feature()
        logger.info("-----Postprocessing data----- ")
        self._post_processing_data()
        # logger.info("-----Save as : {}".format("/".join([config.FEATURE_STORE_ROOT, self.file_name])))
        logger.info("-----Save as : {}".format(config.FeatureData.ORG_FEATURE_DATA.value))
        # self.dump_dataset("/".join([config.FEATURE_STORE_ROOT, self.file_name]))
        self.dump_dataset(config.FeatureData.INTRO_FEATURE_DATA.value)
        logger.info("-----Dataset saved successfully-----")

if __name__ == "__main__":
    # Example usage
    running_dt_end = "2024-06-01"
    train_interval = 100
    file_type = "csv"



