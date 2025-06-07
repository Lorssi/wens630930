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


class RulBaselineFeature(BaseFeatureDataSet):

    def __init__(self, running_dt_end: str, train_interval: int, file_type: str, **param):
        super().__init__(param)
        logger.info('-----Loading data-----')
        self.piglet_data = pd.read_csv(RawData.PIG_LET_DATA.value, encoding='utf-8-sig')
        self.tame_risk_data = pd.read_csv(RawData.ADS_PIG_ISOLATION_TAME_PROLINE_RISK.value, encoding='utf-8-sig', low_memory=False)
        self.index_data = pd.read_csv(RawData.ADS_PIG_ORG_TOTAL_TO_ML_TRAINING_DAY.value, encoding='utf-8-sig')

        self.running_dt_end = running_dt_end
        self.train_interval = train_interval
        self.file_type = file_type

        self.end_date = (datetime.strptime(self.running_dt_end, "%Y-%m-%d") - timedelta(days=1)).strftime("%Y-%m-%d")
        self.start_date = (datetime.strptime(self.end_date, "%Y-%m-%d") - timedelta(days=self.train_interval) - timedelta(days=25)).strftime("%Y-%m-%d")
        logger.info('-----start_date: {}'.format(self.start_date))
        logger.info('-----end_date: {}'.format(self.end_date))

        self.data = pd.DataFrame()
        self.file_name = None

        self._init_entity_and_features()

    def _preprocessing_data(self):

        piglets_data = self.piglet_data.copy()
        piglets_data['stats_dt'] = pd.to_datetime(piglets_data['stats_dt'], format='mixed')
        piglets_data = piglets_data[(piglets_data['stats_dt'] >= self.start_date) & (piglets_data['stats_dt'] <= self.end_date)]

        tame_risk_data = self.tame_risk_data.copy()
        tame_risk_data['allot_dt'] = pd.to_datetime(tame_risk_data['allot_dt'], format='mixed')
        tame_risk_data['min_boar_inpop_dt'] = pd.to_datetime(tame_risk_data['min_boar_inpop_dt'], format='mixed')
        tame_risk_data = tame_risk_data[(tame_risk_data['min_boar_inpop_dt'] >= self.start_date) & (tame_risk_data['min_boar_inpop_dt'] <= self.end_date)]

        index_data = self.index_data.copy()
        index_data['stats_dt'] = pd.to_datetime(index_data['stats_dt'], format='mixed')
        index_data = index_data[(index_data['stats_dt'] >= self.start_date) & (index_data['stats_dt'] <= self.end_date)]
        
        # 更新
        self.piglet_data = piglets_data
        self.tame_risk_data = tame_risk_data
        self.index_data = index_data

    def _init_entity_and_features(self):
        """初始化实体和特征定义"""
        pass

    def _get_check_out_3_feature(self):
        """获取组织特征"""
        data = self.data.copy()
        index_data = self.index_data.copy()
        tame_risk_data = self.tame_risk_data.copy()

        # 先合并数据
        merged_data = pd.merge(
            index_data[['stats_dt', 'pigfarm_dk']], 
            tame_risk_data[['min_boar_inpop_dt', 'prorg_inv_dk', 
                           'rqbe3_blue_ear_kyyd_check_out_qty', 'rqbe3_blue_ear_kypt_check_out_qty']], 
            left_on=['stats_dt', 'pigfarm_dk'], 
            right_on=['min_boar_inpop_dt', 'prorg_inv_dk'], 
            how='left'
        )
        
        # 按日期和猪场分组，计算是否有大于0的值
        grouped = merged_data.groupby(['stats_dt', 'pigfarm_dk']).agg({
            'rqbe3_blue_ear_kyyd_check_out_qty': 'max',
            'rqbe3_blue_ear_kypt_check_out_qty': 'max'
        }).reset_index()
        
        # 计算tame_rule - 只要任一字段大于0即为1
        grouped['tame_rule'] = ((grouped['rqbe3_blue_ear_kyyd_check_out_qty'] > 0) | 
                                (grouped['rqbe3_blue_ear_kypt_check_out_qty'] > 0)).astype(int)
        
        # 只保留需要的列
        result = grouped[['stats_dt', 'pigfarm_dk', 'tame_rule']]
        
        # 与原始数据合并
        if data.empty:
            self.data = result
        else:
            self.data = pd.merge(data, result, on=['stats_dt', 'pigfarm_dk'], how='left')
        
        return self.data
    
    def _get_check_out_3_21d_feature(self):
        """获取近21天是否有蓝耳检测值的特征"""
        data = self.data.copy()
        
        # 确保日期格式正确
        if not pd.api.types.is_datetime64_dtype(data['stats_dt']):
            data['stats_dt'] = pd.to_datetime(data['stats_dt'])
        
        # 创建结果DataFrame
        result_df = data[['stats_dt', 'pigfarm_dk', 'tame_rule']].copy()
        result_df['tame_rule_21d'] = 0  # 默认值为0
        
        # 对每个猪场的数据单独处理
        farm_groups = []
        for farm, group in result_df.groupby('pigfarm_dk'):
            # 确保按日期排序
            group = group.sort_values('stats_dt')
            
            # 计算每个日期前21天内tame_rule最大值
            # 先设置日期为索引以使用rolling函数
            group = group.set_index('stats_dt')
            
            # 向前计算21天滚动窗口的最大值
            group['tame_rule_21d'] = group['tame_rule'].rolling('21d', min_periods=1).max().fillna(0).astype(int)
            
            # 重置索引
            group = group.reset_index()
            farm_groups.append(group)
        
        # 合并所有猪场的结果
        if farm_groups:
            result_df = pd.concat(farm_groups)
            
            # 将结果合并回原始数据
            data = pd.merge(
                data.drop(columns=['tame_rule_21d'], errors='ignore'),
                result_df[['stats_dt', 'pigfarm_dk', 'tame_rule_21d']], 
                on=['stats_dt', 'pigfarm_dk'], 
                how='left'
            )
        
        self.data = data

    def _get_piglet_overstock_feature(self):
        """获取仔猪超存栏特征"""
        data = self.data.copy()
        piglet_data = self.piglet_data.copy()
        
        # 确保日期格式正确
        if not pd.api.types.is_datetime64_dtype(data['stats_dt']):
            data['stats_dt'] = pd.to_datetime(data['stats_dt'])
        if not pd.api.types.is_datetime64_dtype(piglet_data['stats_dt']):
            piglet_data['stats_dt'] = pd.to_datetime(piglet_data['stats_dt'])
        
        # 创建条件标记
        piglet_data['condition_met'] = ((piglet_data['pd03010103'] > 40) & 
                                       (piglet_data['pd25010316'] > 100)).astype(int)
        
        # 按日期和猪场分组，使用max计算是否有满足条件的记录
        grouped = piglet_data.groupby(['stats_dt', 'org_farm_dk']).agg({
            'condition_met': 'max'  # 如果有任一记录满足条件，max值为1
        }).reset_index()
        
        # 重命名列
        grouped = grouped.rename(columns={'condition_met': 'piglet_rule'})
        
        # 与原始数据合并
        result = pd.merge(
            data,
            grouped[['stats_dt', 'org_farm_dk', 'piglet_rule']],
            left_on=['stats_dt', 'pigfarm_dk'],
            right_on=['stats_dt', 'org_farm_dk'],
            how='left'
        )
        
        # 删除多余的列
        if 'org_farm_dk' in result.columns:
            result = result.drop(columns=['org_farm_dk'])
        
        # 填充NaN值为0（当没有匹配记录时）
        result['piglet_rule'] = result['piglet_rule'].fillna(0).astype(int)
        
        self.data = result

    def _get_piglet_overstock_21d_feature(self):
        """获取近21天是否有仔猪超存栏特征"""
        data = self.data.copy()
    
        # 确保日期格式正确
        if not pd.api.types.is_datetime64_dtype(data['stats_dt']):
            data['stats_dt'] = pd.to_datetime(data['stats_dt'])
    
        # 创建结果DataFrame
        result_df = data[['stats_dt', 'pigfarm_dk', 'piglet_rule']].copy()
        result_df['piglet_rule_21d'] = 0  # 默认值为0
    
        # 对每个猪场的数据单独处理
        farm_groups = []
        for farm, group in result_df.groupby('pigfarm_dk'):
            # 确保按日期排序
            group = group.sort_values('stats_dt')
            
            # 计算每个日期前21天内piglet_rule最大值
            # 先设置日期为索引以使用rolling函数
            group = group.set_index('stats_dt')
            
            # 向前计算21天滚动窗口的最大值
            group['piglet_rule_21d'] = group['piglet_rule'].rolling('21d', min_periods=1).max().fillna(0).astype(int)
            
            # 重置索引
            group = group.reset_index()
            farm_groups.append(group)
    
        # 合并所有猪场的结果
        if farm_groups:
            result_df = pd.concat(farm_groups)
            
            # 将结果合并回原始数据
            data = pd.merge(
                data.drop(columns=['piglet_rule_21d'], errors='ignore'),
                result_df[['stats_dt', 'pigfarm_dk', 'piglet_rule_21d']], 
                on=['stats_dt', 'pigfarm_dk'], 
                how='left'
            )
    
        self.data = data

    def _get_pig_check_out_8_30_feature(self):
        """获取蓝耳检出特征(intro_rule)"""
        data = self.data.copy()
        tame_risk_data = self.tame_risk_data.copy()
        
        # 确保日期格式正确
        if not pd.api.types.is_datetime64_dtype(data['stats_dt']):
            data['stats_dt'] = pd.to_datetime(data['stats_dt'])
        if not pd.api.types.is_datetime64_dtype(tame_risk_data['allot_dt']):
            tame_risk_data['allot_dt'] = pd.to_datetime(tame_risk_data['allot_dt'])
        
        # 先合并数据
        merged_data = pd.merge(
            data[['stats_dt', 'pigfarm_dk']], 
            tame_risk_data[['allot_dt', 'prorg_inv_dk', 'yzbe8_blue_ear_kyyd_check_out_qty']], 
            left_on=['stats_dt', 'pigfarm_dk'], 
            right_on=['allot_dt', 'prorg_inv_dk'], 
            how='left'
        )
        
        # 按日期和猪场分组，计算是否有大于0的值
        grouped = merged_data.groupby(['stats_dt', 'pigfarm_dk']).agg({
            'yzbe8_blue_ear_kyyd_check_out_qty': 'max'
        }).reset_index()
        
        # 计算intro_rule - 值大于0即为1
        grouped['intro_rule'] = (grouped['yzbe8_blue_ear_kyyd_check_out_qty'] > 0).astype(int)
        
        # 只保留需要的列
        result = grouped[['stats_dt', 'pigfarm_dk', 'intro_rule']]
        
        # 与原始数据合并
        data = pd.merge(
            data.drop(columns=['intro_rule'], errors='ignore'),
            result, 
            on=['stats_dt', 'pigfarm_dk'], 
            how='left'
        )
        
        # 填充NaN值为0（当没有匹配记录时）
        data['intro_rule'] = data['intro_rule'].fillna(0).astype(int)
        self.data = data

    def _get_pig_check_out_8_30_21d_feature(self):
        """获取近21天是否有蓝耳检出特征(intro_rule_21d)"""
        data = self.data.copy()
    
        # 确保日期格式正确
        if not pd.api.types.is_datetime64_dtype(data['stats_dt']):
            data['stats_dt'] = pd.to_datetime(data['stats_dt'])
    
        # 创建结果DataFrame
        result_df = data[['stats_dt', 'pigfarm_dk', 'intro_rule']].copy()
        result_df['intro_rule_21d'] = 0  # 默认值为0
    
        # 对每个猪场的数据单独处理
        farm_groups = []
        for farm, group in result_df.groupby('pigfarm_dk'):
            # 确保按日期排序
            group = group.sort_values('stats_dt')
            
            # 计算每个日期前21天内intro_rule最大值
            # 先设置日期为索引以使用rolling函数
            group = group.set_index('stats_dt')
            
            # 向前计算21天滚动窗口的最大值
            group['intro_rule_21d'] = group['intro_rule'].rolling('21d', min_periods=1).max().fillna(0).astype(int)
            
            # 重置索引
            group = group.reset_index()
            farm_groups.append(group)
    
        # 合并所有猪场的结果
        if farm_groups:
            result_df = pd.concat(farm_groups)
            
            # 将结果合并回原始数据
            data = pd.merge(
                data.drop(columns=['intro_rule_21d'], errors='ignore'),
                result_df[['stats_dt', 'pigfarm_dk', 'intro_rule_21d']], 
                on=['stats_dt', 'pigfarm_dk'], 
                how='left'
            )
    
        self.data = data

    def _post_processing_data(self):
        if self.data.isnull().any().any():
            logger.info("Warning: Null in org_feature_data.csv")
        self.file_name = "org_feature_data." + self.file_type

        data = self.data.copy()
        data['stats_dt'] = pd.to_datetime(data['stats_dt'])
        data['stats_dt'] = data['stats_dt'] + pd.DateOffset(days=1)  # 将日期加1天
        self.data = data

    def build_dataset_all(self):
        logger.info("-----Preprocessing data----- ")
        self._preprocessing_data()
        logger.info("Calculating check_out_3 purchase...")
        self._get_check_out_3_feature()
        logger.info("Calculating check_out_3 21 days purchase...")
        self._get_check_out_3_21d_feature()
        logger.info("Calculating check_out_3 21 days purchase...")
        self._get_check_out_3_21d_feature()
        logger.info("Calculating piglet_overstock purchase...")
        self._get_piglet_overstock_feature()
        logger.info("Calculating piglet_overstock 21 days purchase...")
        self._get_piglet_overstock_21d_feature()
        logger.info("Calculating pig_check_out_8_30 purchase...")
        self._get_pig_check_out_8_30_feature()
        logger.info("Calculating pig_check_out_8_30 21 days purchase...")
        self._get_pig_check_out_8_30_21d_feature()
        logger.info("-----Postprocessing data----- ")
        self._post_processing_data()
        # logger.info("-----Save as : {}".format("/".join([config.FEATURE_STORE_ROOT, self.file_name])))
        logger.info("-----Save as : {}".format(config.FeatureData.RULE_BASELINE_FEATURE_DATA.value))
        # self.dump_dataset("/".join([config.FEATURE_STORE_ROOT, self.file_name]))
        self.dump_dataset(config.FeatureData.RULE_BASELINE_FEATURE_DATA.value)
        logger.info("-----Dataset saved successfully-----")

if __name__ == "__main__":
    # Example usage
    running_dt_end = "2024-06-01"
    train_interval = 100
    file_type = "csv"

    dataset = OrgLocationFeature(running_dt_end, train_interval, file_type)
    dataset.build_dataset_all()
    logger.info("Production feature dataset built successfully.")



