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


class PrrsCheckFeature(BaseFeatureDataSet):

    def __init__(self, running_dt_end: str, train_interval: int, file_type: str, **param):
        super().__init__(param)
        logger.info('-----Loading data-----')
        self.production_data = pd.read_csv(RawData.ADS_PIG_ORG_TOTAL_TO_ML_TRAINING_DAY.value, encoding='utf-8-sig')
        self.check_data = pd.read_csv(RawData.TMP_PIG_ORG_DISEASE_CHECK_RESULT_DAY.value, encoding='utf-8-sig')

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
        production_data = self.production_data[['stats_dt', 'pigfarm_dk']].copy()
        check_data = self.check_data[self.entity + ['check_item_dk', 'index_item_dk', 'check_qty', 'check_out_qty']].copy()
        
        production_data['stats_dt'] = pd.to_datetime(production_data['stats_dt'])
        production_data = production_data[(production_data['stats_dt'] >= self.start_date) & (production_data['stats_dt'] <= self.end_date)]
        check_data['receive_dt'] = pd.to_datetime(check_data['receive_dt'])
        check_data = check_data[(check_data['receive_dt'] >= self.start_date) & (check_data['receive_dt'] <= self.end_date)]

        # 过滤掉不需要的检查项
        """
        筛选PRRS数据
        :param check_data: 输入数据
        :return: 筛选后的数据
        """
        # 定义特殊项目ID列表
        special_items = [
            'bDoAA065yHyCrSt1',
            'bDoAAzoDnCKCrSt1',
            'bDoAAvklLFWCrSt1',
        ]
        
        # 定义所有需要的check_item_dk
        all_items = [
            # 野毒
            'bDoAAfRM6YiCrSt1',
            'bDoAArPPgj6CrSt1',
            'bDoAAfRM6IGCrSt1',
            'bDoAAfYsNUGCrSt1',
            'bDoAAfYsM8eCrSt1',
            'bDoAAfYr79SCrSt1',
            # 抗原
            'bDoAAJyZSTSCrSt1',
            'bDoAAfYgkW2CrSt1',
            'bDoAAfYq6LWCrSt1',
            'bDoAAfYq6kWCrSt1',
            'bDoAAfYsNKyCrSt1',
            'bDoAAwWyhPOCrSt1',
            # 抗体
            'bDoAAJyZSZiCrSt1',
            # 特殊项目
            'bDoAA065yHyCrSt1',
            'bDoAAzoDnCKCrSt1',
            'bDoAAvklLFWCrSt1',
        ]
        
        # 定义需要的index_item_dk
        valid_indexes = [
            # 野毒
            'bDoAAfYcdbLWD/D5',
            'bDoAAfYcdbTWD/D5',
            'bDoAAfRPf0jWD/D5',
            'bDoAAKqewlzWD/D5',
            # 条带
            'bDoAAKqffmXWD/D5',
            'bDoAAKqewhjWD/D5',
            # prrsvct
            'bDoAAKqffxXWD/D5',
            # 抗原
            'bDoAAfYq6kvWD/D5',
            # 抗体
            'bDoAAKqZiKzWD/D5',
            # s/p
            'bDoAAKqZiKzWD/D5',
        ]
        
        # 步骤1: 首先按照check_item_dk筛选所有数据
        filtered_data = check_data[check_data['check_item_dk'].isin(all_items)]
        
        # 步骤2: 将数据分为两部分
        # 非特殊项目数据 - 直接保留
        regular_items = filtered_data[~filtered_data['check_item_dk'].isin(special_items)]
        
        # 特殊项目数据 - 需要进一步筛选index_item_dk
        special_items_data = filtered_data[filtered_data['check_item_dk'].isin(special_items)]
        filtered_special_items = special_items_data[special_items_data['index_item_dk'].isin(valid_indexes)]
        
        # 步骤3: 合并两部分数据
        final_data = pd.concat([regular_items, filtered_special_items])
        
        logger.info(f"检测数据筛选蓝耳前数据量: {len(check_data)}, 筛选后数据量: {len(final_data)}")
        
        # 更新
        self.production_data = production_data
        self.check_data = final_data

    def _init_entity_and_features(self):
        """初始化实体和特征定义"""
        # 设置实体列表
        self.entity = ['receive_dt', 'org_inv_dk']
        # 定义特征
        features_config = [
            # 检测特征
            ('check_out_ratio_7d', FeatureType.Continuous, FeatureDtype.Int32, 'check'),
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

    def _get_check_out_7d_feature(self):
        """获取组织特征"""
        production_data = self.production_data.copy()
        check_data = self.check_data.copy()
        check_data = check_data.rename(columns={'receive_dt': 'stats_dt'})
        check_data = check_data.rename(columns={'org_inv_dk': 'pigfarm_dk'})

        """
        计算PRRS检查结果的阳性率，包括总阳性率和野毒阳性率
        对于每个猪场，计算其近7天内的PRRS检测阳性率
        """
        # 定义野毒相关的check_item_dk
        wild_check_items = [
            'bDoAAfRM6YiCrSt1', 
            'bDoAArPPgj6CrSt1', 
            'bDoAAfRM6IGCrSt1', 
            'bDoAAfYsNUGCrSt1', 
            'bDoAAfYsM8eCrSt1', 
            'bDoAAfYr79SCrSt1'
        ]
        
        # 定义特殊项目
        special_items = [
            'bDoAA065yHyCrSt1',
            'bDoAAzoDnCKCrSt1',
            'bDoAAvklLFWCrSt1',
        ]
        
        # 定义野毒相关的index_item_dk
        wild_indexes = [
            'bDoAAfYcdbLWD/D5',
            'bDoAAfYcdbTWD/D5',
            'bDoAAfRPf0jWD/D5',
            'bDoAAKqewlzWD/D5',
        ]
        
        # 确保有数据可处理
        if len(check_data) == 0:
            logger.warning("没有检测数据可供处理")
            # 如果没有结果数据，添加默认列
            production_data['check_out_ratio_7d'] = np.nan
            production_data['wild_check_out_ratio_7d'] = np.nan
            self.data = production_data.copy()
            return
        
        # 为check_data添加标记，标识是否为野毒数据
        # 条件1: check_item_dk在wild_check_items中
        # 条件2: check_item_dk在special_items中且index_item_dk在wild_indexes中
        check_data['is_wild'] = (
            check_data['check_item_dk'].isin(wild_check_items) | 
            ((check_data['check_item_dk'].isin(special_items)) & 
             (check_data['index_item_dk'].isin(wild_indexes)))
        )
        
        # 分别计算每个猪场每天的检测总量和阳性数量
        # 使用groupby优化计算效率
        result_data = []
        
        # 先获取所有唯一的日期和猪场组合
        unique_dates = production_data['stats_dt'].unique()
        unique_farms = production_data['pigfarm_dk'].unique()

        # 预处理数据，提高后续计算效率
        check_data_preprocessed = check_data[['pigfarm_dk', 'stats_dt', 'check_qty', 'check_out_qty', 'is_wild']].copy()
        
        for farm_dk in tqdm(unique_farms):
            farm_check_data = check_data_preprocessed[check_data_preprocessed['pigfarm_dk'] == farm_dk]
            
            for target_date in unique_dates:
                # 计算7天日期范围
                start_date = pd.to_datetime(target_date) - pd.Timedelta(days=6)
                
                # 过滤该日期范围内的数据
                date_range_data = farm_check_data[
                    (farm_check_data['stats_dt'] >= start_date) & 
                    (farm_check_data['stats_dt'] <= target_date)
                ]
                
                if len(date_range_data) == 0:
                    continue
                
                # 计算总阳性率
                total_check_qty = date_range_data['check_qty'].sum()
                total_check_out_qty = date_range_data['check_out_qty'].sum()
                total_positive_rate = total_check_out_qty / total_check_qty if total_check_qty > 0 else np.nan
                
                # 计算野毒阳性率
                wild_data = date_range_data[date_range_data['is_wild']]
                wild_check_qty = wild_data['check_qty'].sum()
                wild_check_out_qty = wild_data['check_out_qty'].sum()
                wild_positive_rate = wild_check_out_qty / wild_check_qty if wild_check_qty > 0 else np.nan
                
                result_data.append({
                    'pigfarm_dk': farm_dk,
                    'stats_dt': target_date,
                    'prrs_7d_positive_rate': round(total_positive_rate, 4),
                    'prrs_wild_7d_positive_rate': round(wild_positive_rate, 4)
                })
        
        
        # 将结果转换为DataFrame并与index_data合并
        if result_data:
            result_df = pd.DataFrame(result_data)

            result_df['stats_dt'] = pd.to_datetime(result_df['stats_dt'])
            # result_df['stats_dt'] = result_df['stats_dt'] + pd.DateOffset(days=1)  # 将日期加1天，确保预警运行边界日期的特征为观察期窗口特征
            
            production_data = production_data.merge(
                result_df, 
                on=['pigfarm_dk', 'stats_dt'], 
                how='left'
            )
            
            # 填充空值
            production_data['check_out_ratio_7d'] = production_data['prrs_7d_positive_rate']
            production_data['wild_check_out_ratio_7d'] = production_data['prrs_wild_7d_positive_rate']
            self.data = production_data[['stats_dt', 'pigfarm_dk', 'check_out_ratio_7d', 'wild_check_out_ratio_7d']].copy()
        else:
            # 如果没有结果数据，添加默认列
            production_data['check_out_ratio_7d'] = np.nan
            production_data['wild_check_out_ratio_7d'] = np.nan

            self.data = production_data[['stats_dt', 'pigfarm_dk', 'check_out_ratio_7d', 'wild_check_out_ratio_7d']].copy()
        
        logger.info(f"计算完成PRRS 7天阳性率和野毒阳性率，总数据量: {len(self.data)}")

    def _get_check_out_15d_feature(self):
        """获取组织特征"""
        production_data = self.production_data.copy()
        check_data = self.check_data.copy()
        check_data = check_data.rename(columns={'receive_dt': 'stats_dt'})
        check_data = check_data.rename(columns={'org_inv_dk': 'pigfarm_dk'})

        """
        计算PRRS检查结果的阳性率，包括总阳性率和野毒阳性率
        对于每个猪场，计算其近15天内的PRRS检测阳性率
        """
        # 定义野毒相关的check_item_dk
        wild_check_items = [
            'bDoAAfRM6YiCrSt1', 
            'bDoAArPPgj6CrSt1', 
            'bDoAAfRM6IGCrSt1', 
            'bDoAAfYsNUGCrSt1', 
            'bDoAAfYsM8eCrSt1', 
            'bDoAAfYr79SCrSt1'
        ]
        
        # 定义特殊项目
        special_items = [
            'bDoAA065yHyCrSt1',
            'bDoAAzoDnCKCrSt1',
            'bDoAAvklLFWCrSt1',
        ]
        
        # 定义野毒相关的index_item_dk
        wild_indexes = [
            'bDoAAfYcdbLWD/D5',
            'bDoAAfYcdbTWD/D5',
            'bDoAAfRPf0jWD/D5',
            'bDoAAKqewlzWD/D5',
        ]
        
        # 确保有数据可处理
        if len(check_data) == 0:
            logger.warning("没有检测数据可供处理")
            # 如果没有结果数据，添加默认列
            production_data['check_out_ratio_15d'] = np.nan
            production_data['wild_check_out_ratio_15d'] = np.nan
            self.data =  pd.merge(
                self.data,
                production_data[['stats_dt', 'pigfarm_dk', 'check_out_ratio_15d', 'wild_check_out_ratio_15d']].copy(),
                on=["stats_dt", "pigfarm_dk"],
                how="left" 
            )
            return
        
        # 为check_data添加标记，标识是否为野毒数据
        # 条件1: check_item_dk在wild_check_items中
        # 条件2: check_item_dk在special_items中且index_item_dk在wild_indexes中
        check_data['is_wild'] = (
            check_data['check_item_dk'].isin(wild_check_items) | 
            ((check_data['check_item_dk'].isin(special_items)) & 
             (check_data['index_item_dk'].isin(wild_indexes)))
        )
        
        # 分别计算每个猪场每天的检测总量和阳性数量
        # 使用groupby优化计算效率
        result_data = []
        
        # 先获取所有唯一的日期和猪场组合
        unique_dates = production_data['stats_dt'].unique()
        unique_farms = production_data['pigfarm_dk'].unique()
        
        # 预处理数据，提高后续计算效率
        check_data_preprocessed = check_data[['pigfarm_dk', 'stats_dt', 'check_qty', 'check_out_qty', 'is_wild']].copy()
        
        for farm_dk in tqdm(unique_farms):
            farm_check_data = check_data_preprocessed[check_data_preprocessed['pigfarm_dk'] == farm_dk]
            
            for target_date in unique_dates:
                # 计算15天日期范围
                start_date = pd.to_datetime(target_date) - pd.Timedelta(days=14)

                # 过滤该日期范围内的数据
                date_range_data = farm_check_data[
                    (farm_check_data['stats_dt'] >= start_date) & 
                    (farm_check_data['stats_dt'] <= target_date)
                ]
                
                if len(date_range_data) == 0:
                    continue
                
                # 计算总阳性率
                total_check_qty = date_range_data['check_qty'].sum()
                total_check_out_qty = date_range_data['check_out_qty'].sum()
                total_positive_rate = total_check_out_qty / total_check_qty if total_check_qty > 0 else np.nan
                
                # 计算野毒阳性率
                wild_data = date_range_data[date_range_data['is_wild']]
                wild_check_qty = wild_data['check_qty'].sum()
                wild_check_out_qty = wild_data['check_out_qty'].sum()
                wild_positive_rate = wild_check_out_qty / wild_check_qty if wild_check_qty > 0 else np.nan
                
                result_data.append({
                    'pigfarm_dk': farm_dk,
                    'stats_dt': target_date,
                    'prrs_15d_positive_rate': round(total_positive_rate, 4),
                    'prrs_wild_15d_positive_rate': round(wild_positive_rate, 4)
                })
        
        
        # 将结果转换为DataFrame并与index_data合并
        if result_data:
            result_df = pd.DataFrame(result_data)

            result_df['stats_dt'] = pd.to_datetime(result_df['stats_dt'])
            # result_df['stats_dt'] = result_df['stats_dt'] + pd.DateOffset(days=1)  # 将日期加1天，确保预警运行边界日期的特征为观察期窗口特征
            
            production_data = production_data.merge(
                result_df, 
                on=['pigfarm_dk', 'stats_dt'], 
                how='left'
            )
            
            # 填充空值
            production_data['check_out_ratio_15d'] = production_data['prrs_15d_positive_rate']
            production_data['wild_check_out_ratio_15d'] = production_data['prrs_wild_15d_positive_rate']
            self.data =  pd.merge(
                self.data,
                production_data[['stats_dt', 'pigfarm_dk', 'check_out_ratio_15d', 'wild_check_out_ratio_15d']].copy(),
                on=["stats_dt", "pigfarm_dk"],
                how="left" 
            )
        else:
            # 如果没有结果数据，添加默认列
            production_data['check_out_ratio_15d'] = np.nan
            production_data['wild_check_out_ratio_15d'] = np.nan
            self.data =  pd.merge(
                self.data,
                production_data[['stats_dt', 'pigfarm_dk', 'check_out_ratio_15d', 'wild_check_out_ratio_15d']].copy(),
                on=["stats_dt", "pigfarm_dk"],
                how="left" 
            )

        logger.info(f"计算完成PRRS 15天阳性率和野毒阳性率，总数据量: {len(self.data)}")

    def _get_check_out_feature(self, offset = 7):
        """获取组织特征"""
        production_data = self.production_data.copy()
        check_data = self.check_data.copy()
        check_data = check_data.rename(columns={'receive_dt': 'stats_dt'})
        check_data = check_data.rename(columns={'org_inv_dk': 'pigfarm_dk'})

        """
        计算PRRS检查结果的阳性率，包括总阳性率和野毒阳性率
        对于每个猪场，计算其近15天内的PRRS检测阳性率
        """
        # 定义野毒相关的check_item_dk
        wild_check_items = [
            'bDoAAfRM6YiCrSt1', 
            'bDoAArPPgj6CrSt1', 
            'bDoAAfRM6IGCrSt1', 
            'bDoAAfYsNUGCrSt1', 
            'bDoAAfYsM8eCrSt1', 
            'bDoAAfYr79SCrSt1'
        ]
        
        # 定义特殊项目
        special_items = [
            'bDoAA065yHyCrSt1',
            'bDoAAzoDnCKCrSt1',
            'bDoAAvklLFWCrSt1',
        ]
        
        # 定义野毒相关的index_item_dk
        wild_indexes = [
            'bDoAAfYcdbLWD/D5',
            'bDoAAfYcdbTWD/D5',
            'bDoAAfRPf0jWD/D5',
            'bDoAAKqewlzWD/D5',
        ]
        
        # 确保有数据可处理
        if len(check_data) == 0:
            logger.warning("没有检测数据可供处理")
            # 如果没有结果数据，添加默认列
            production_data[f'check_out_ratio_{offset}d'] = np.nan
            production_data[f'wild_check_out_ratio_{offset}d'] = np.nan
            if self.data.empty:
                self.data = production_data[['stats_dt', 'pigfarm_dk', f'check_out_ratio_{offset}d', f'wild_check_out_ratio_{offset}d']].copy()
            else:
                self.data =  pd.merge(
                    self.data,
                    production_data[['stats_dt', 'pigfarm_dk', f'check_out_ratio_{offset}d', f'wild_check_out_ratio_{offset}d']].copy(),
                    on=["stats_dt", "pigfarm_dk"],
                    how="left" 
                )
            return
        
        # 为check_data添加标记，标识是否为野毒数据
        # 条件1: check_item_dk在wild_check_items中
        # 条件2: check_item_dk在special_items中且index_item_dk在wild_indexes中
        check_data['is_wild'] = (
            check_data['check_item_dk'].isin(wild_check_items) | 
            ((check_data['check_item_dk'].isin(special_items)) & 
             (check_data['index_item_dk'].isin(wild_indexes)))
        )
        
        # 分别计算每个猪场每天的检测总量和阳性数量
        # 使用groupby优化计算效率
        result_data = []
        
        # 先获取所有唯一的日期和猪场组合
        unique_dates = production_data['stats_dt'].unique()
        unique_farms = production_data['pigfarm_dk'].unique()
        
        # 预处理数据，提高后续计算效率
        check_data_preprocessed = check_data[['pigfarm_dk', 'stats_dt', 'check_qty', 'check_out_qty', 'is_wild']].copy()
        
        for farm_dk in tqdm(unique_farms):
            farm_check_data = check_data_preprocessed[check_data_preprocessed['pigfarm_dk'] == farm_dk]
            
            for target_date in unique_dates:
                # 计算15天日期范围
                start_date = pd.to_datetime(target_date) - pd.Timedelta(days=offset - 1)

                # 过滤该日期范围内的数据
                date_range_data = farm_check_data[
                    (farm_check_data['stats_dt'] >= start_date) & 
                    (farm_check_data['stats_dt'] <= target_date)
                ]
                
                if len(date_range_data) == 0:
                    continue
                
                # 计算总阳性率
                total_check_qty = date_range_data['check_qty'].sum()
                total_check_out_qty = date_range_data['check_out_qty'].sum()
                total_positive_rate = total_check_out_qty / total_check_qty if total_check_qty > 0 else np.nan
                
                # 计算野毒阳性率
                wild_data = date_range_data[date_range_data['is_wild']]
                wild_check_qty = wild_data['check_qty'].sum()
                wild_check_out_qty = wild_data['check_out_qty'].sum()
                wild_positive_rate = wild_check_out_qty / wild_check_qty if wild_check_qty > 0 else np.nan
                
                result_data.append({
                    'pigfarm_dk': farm_dk,
                    'stats_dt': target_date,
                    f'prrs_{offset}d_positive_rate': round(total_positive_rate, 4),
                    f'prrs_wild_{offset}d_positive_rate': round(wild_positive_rate, 4)
                })
        
        
        # 将结果转换为DataFrame并与index_data合并
        if result_data:
            result_df = pd.DataFrame(result_data)

            result_df['stats_dt'] = pd.to_datetime(result_df['stats_dt'])
            
            production_data = production_data.merge(
                result_df, 
                on=['pigfarm_dk', 'stats_dt'], 
                how='left'
            )
            
            # 填充空值
            production_data[f'check_out_ratio_{offset}d'] = production_data[f'prrs_{offset}d_positive_rate']
            production_data[f'wild_check_out_ratio_{offset}d'] = production_data[f'prrs_wild_{offset}d_positive_rate']
            
            # 检查self.data是否为空
            if self.data.empty:
                self.data = production_data[['stats_dt', 'pigfarm_dk', f'check_out_ratio_{offset}d', f'wild_check_out_ratio_{offset}d']].copy()
            else:
                self.data = pd.merge(
                    self.data,
                    production_data[['stats_dt', 'pigfarm_dk', f'check_out_ratio_{offset}d', f'wild_check_out_ratio_{offset}d']].copy(),
                    on=["stats_dt", "pigfarm_dk"],
                    how="left" 
                )
        else:
            # 如果没有结果数据，添加默认列
            production_data[f'check_out_ratio_{offset}d'] = np.nan
            production_data[f'wild_check_out_ratio_{offset}d'] = np.nan
            
            # 检查self.data是否为空
            if self.data.empty:
                self.data = production_data[['stats_dt', 'pigfarm_dk', f'check_out_ratio_{offset}d', f'wild_check_out_ratio_{offset}d']].copy()
            else:
                self.data = pd.merge(
                    self.data,
                    production_data[['stats_dt', 'pigfarm_dk', f'check_out_ratio_{offset}d', f'wild_check_out_ratio_{offset}d']].copy(),
                    on=["stats_dt", "pigfarm_dk"],
                    how="left" 
                )

        logger.info(f"计算完成PRRS {offset}天阳性率和野毒阳性率，总数据量: {len(self.data)}")

    def _post_processing_data(self):
        if self.data.isnull().any().any():
            logger.info("Warning: Null in check_feature_data.csv")
        self.file_name = "check_feature_data." + self.file_type

        data = self.data.copy()
        data['stats_dt'] = pd.to_datetime(data['stats_dt'])
        data['stats_dt'] = data['stats_dt'] + pd.DateOffset(days=1)
        self.data = data.copy()

    def build_dataset_all(self):
        logger.info("-----Preprocessing data----- ")
        self._preprocessing_data()
        logger.info("Calculating interval from last purchase...")
        self._get_check_out_feature(offset=7)
        logger.info("-----Postprocessing data----- ")
        self._post_processing_data()
        # logger.info("-----Save as : {}".format("/".join([config.FEATURE_STORE_ROOT, self.file_name])))
        logger.info("-----Save as : {}".format(config.FeatureData.CHECK_FEATURE_DATA.value))
        # self.dump_dataset("/".join([config.FEATURE_STORE_ROOT, self.file_name]))
        self.dump_dataset(config.FeatureData.CHECK_FEATURE_DATA.value)
        self.data.to_csv('data.csv', index=False, encoding='utf-8-sig')

if __name__ == "__main__":
    # Example usage
    running_dt_end = "2024-06-01"
    train_interval = 100
    file_type = "csv"



