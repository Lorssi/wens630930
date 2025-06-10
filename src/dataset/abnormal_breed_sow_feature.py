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


class AbnormalBreedSowFeature(BaseFeatureDataSet):

    def __init__(self, running_dt_end: str, train_interval: int, file_type: str, **param):
        super().__init__(param)
        logger.info('-----Loading data-----')
        self.abnormal_data = pd.read_csv(RawData.ABNORMAL_BOAR_REPORT_MODEL_DATA.value, encoding='utf-8-sig')
        self.index_data = pd.read_csv(RawData.ADS_PIG_ORG_TOTAL_TO_ML_TRAINING_DAY.value, encoding='utf-8-sig')

        self.running_dt_end = running_dt_end
        self.train_interval = train_interval
        self.file_type = file_type

        self.end_date = (datetime.strptime(self.running_dt_end, "%Y-%m-%d") - timedelta(days=1)).strftime("%Y-%m-%d")
        self.start_date = (datetime.strptime(self.end_date, "%Y-%m-%d") - timedelta(days=self.train_interval) - timedelta(days=20)).strftime("%Y-%m-%d")
        logger.info('-----start_date: {}'.format(self.start_date))
        logger.info('-----end_date: {}'.format(self.end_date))

        self.data = pd.DataFrame()
        self.file_name = None

        self._init_entity_and_features()

    def _preprocessing_data(self):
        """预处理数据"""
        index_data = self.index_data.copy()
        abnormal_data = self.abnormal_data.copy()
        # 过滤日期范围
        index_data['stats_dt'] = pd.to_datetime(index_data['stats_dt'])
        index_data = index_data[(index_data['stats_dt'] >= self.start_date) & (index_data['stats_dt'] <= self.end_date)]
        index_data = index_data[['stats_dt', 'pigfarm_dk']].reset_index(drop=True)

        # abnormal_data = abnormal_data[abnormal_data['abnormal_type_nm'] == '流产']
        
        # 更新
        self.index_data = index_data.copy()
        self.abnormal_data = abnormal_data.copy()

    def _init_entity_and_features(self):
        """初始化实体和特征定义"""
        # 设置实体列表
        self.entity = ['abnormal_discv_tm', 'org_inv_dk']
        # 定义特征
        features_config = [
            # 组织特征
            ('abnormal_find_days', FeatureType.Continuous, FeatureDtype.Int32, 'abnormal_discv'),
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

    def _get_abnormal_find_days_feature(self):
        """使用分组操作计算每个猪场最近一次异常距离当前日期的天数"""
        index_data = self.index_data.copy()
        abnormal_data = self.abnormal_data.copy()
        data = self.data.copy()
        
        # 确保日期列的格式正确
        index_data['stats_dt'] = pd.to_datetime(index_data['stats_dt'])
        abnormal_data['abnormal_discv_tm'] = pd.to_datetime(abnormal_data['abnormal_discv_tm'])
        
        # 创建结果DataFrame
        result_df = index_data.copy()
        
        # 按猪场分组处理异常数据
        abnormal_dates = {}
        
        # 对每个猪场的异常日期进行预处理和排序
        for farm, group in abnormal_data.groupby('org_inv_dk'):
            # 获取该猪场所有异常日期并排序
            farm_abnormal_dates = sorted(group['abnormal_discv_tm'].unique())
            if farm_abnormal_dates:
                abnormal_dates[farm] = farm_abnormal_dates
        
        # 向量化查找最近的异常日期
        def find_closest_date(row):
            farm = row['pigfarm_dk']
            date = row['stats_dt']
            
            if farm not in abnormal_dates:
                return np.nan
            
            # 二分查找最接近但小于当前日期的异常日期
            dates = abnormal_dates[farm]
            idx = np.searchsorted(dates, date) - 1  # 找到最近的之前日期的索引
            
            if idx >= 0:
                closest_date = dates[idx]
                return (date - closest_date).days
            return np.nan
        
        # 使用向量化操作代替行级apply
        result_df['abnormal_find_days'] = result_df.apply(find_closest_date, axis=1)
        
        # 选择需要的列
        result_df = result_df[['stats_dt', 'pigfarm_dk', 'abnormal_find_days']]
        
        # 更新数据集
        if data.empty:
            data = result_df
        else:
            # 合并到现有数据
            data = pd.merge(
                data,
                result_df,
                on=['stats_dt', 'pigfarm_dk'],
                how='left'
            )

        self.data = data.copy()
        logger.info(f"已计算最近一次异常距离当前日期的天数 - 记录数量: {len(result_df)}")
        
    def _get_abnormal_boar_num_15d_feature(self):
        """计算每个猪场在最近15天内的异常猪数量"""
        index_data = self.index_data.copy()
        abnormal_data = self.abnormal_data.copy()
        data = self.data.copy()
        
        # 确保日期列的格式正确
        index_data['stats_dt'] = pd.to_datetime(index_data['stats_dt'])
        abnormal_data['abnormal_discv_tm'] = pd.to_datetime(abnormal_data['abnormal_discv_tm'])
        
        # 创建结果DataFrame
        result_df = index_data.copy()
        
        # 预处理：按猪场分组并提前获取每个猪场的异常数据
        farm_abnormal_dict = {}
        for farm, group in abnormal_data.groupby('org_inv_dk'):
            # 保存猪场的异常数据：日期和猪耳号
            farm_abnormal_dict[farm] = group[['abnormal_discv_tm', 'ear_no']].copy()
        
        # 定义函数计算15天窗口内的异常猪数量
        def count_abnormal_in_15d(row):
            farm = row['pigfarm_dk']
            date = row['stats_dt']
            
            if farm not in farm_abnormal_dict:
                return 0
            
            # 获取该猪场的异常数据
            farm_data = farm_abnormal_dict[farm]
            
            # 计算15天窗口的开始日期
            start_date = date - pd.Timedelta(days=14)  # 15天窗口包括当天
            
            # 筛选窗口内的异常记录
            window_data = farm_data[(farm_data['abnormal_discv_tm'] >= start_date) & 
                                 (farm_data['abnormal_discv_tm'] <= date)]
            
            # 统计唯一的异常猪数量
            return window_data['ear_no'].nunique()
        
        # 应用函数计算15天窗口内的异常猪数量
        result_df['abnormal_abort_boar_num_15d'] = result_df.apply(count_abnormal_in_15d, axis=1)
        
        # 选择需要的列
        result_df = result_df[['stats_dt', 'pigfarm_dk', 'abnormal_abort_boar_num_15d']]
        
        # 更新数据集
        if data.empty:
            data = result_df
        else:
            # 合并到现有数据
            data = pd.merge(
                data,
                result_df,
                on=['stats_dt', 'pigfarm_dk'],
                how='left'
            )
    
        self.data = data.copy()
        logger.info(f"已计算15天内异常猪数量 - 记录数量: {len(result_df)}")

    def _post_processing_data(self):
        if self.data.isnull().any().any():
            logger.info("Warning: Null in org_feature_data.csv")
        self.file_name = "abnormal_boar_feature_data." + self.file_type

        data = self.data.copy()
        data['stats_dt'] = pd.to_datetime(data['stats_dt'])
        data['stats_dt'] = data['stats_dt'] + pd.DateOffset(days=1)  # 确保日期是正确的
        self.data = data.copy()

    def build_dataset_all(self):
        logger.info("-----Preprocessing data----- ")
        self._preprocessing_data()
        logger.info("Calculating interval from last purchase...")
        self._get_abnormal_find_days_feature()
        # self._get_abnormal_boar_num_15d_feature()
        logger.info("-----Postprocessing data----- ")
        self._post_processing_data()
        # logger.info("-----Save as : {}".format("/".join([config.FEATURE_STORE_ROOT, self.file_name])))
        logger.info("-----Save as : {}".format(config.FeatureData.ABNORMAL_BOAR_FEATURE_DATA.value))
        # self.dump_dataset("/".join([config.FEATURE_STORE_ROOT, self.file_name]))
        self.dump_dataset(config.FeatureData.ABNORMAL_BOAR_FEATURE_DATA.value)
        logger.info("-----Dataset saved successfully-----")

if __name__ == "__main__":
    # Example usage
    running_dt_end = "2024-06-01"
    train_interval = 100
    file_type = "csv"

    dataset = OrgLocationFeature(running_dt_end, train_interval, file_type)
    dataset.build_dataset_all()
    logger.info("Production feature dataset built successfully.")



