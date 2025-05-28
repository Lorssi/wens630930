# data_loader/preprocessing.py
import pandas as pd
import numpy as np
from utils.logger import setup_logger
from configs.logger_config import logger_config
from configs.feature_config import DataPathConfig, ColumnsConfig
from tqdm import tqdm
logger = setup_logger(logger_config.TRAIN_LOG_FILE_PATH, logger_name="PreprocessingLogger")

class CheckPreprocessor:
    def __init__(self, index_data, running_dt="2024-10-01", interval_days=400):
        self.index_data = index_data.copy()  # 深拷贝，避免对原数据的修改
        self.check_data = pd.read_csv(DataPathConfig.PRRS_CHECK_DATA_PATH, encoding='utf-8')
        self.calculate_data = None

        self.start_date = pd.to_datetime(running_dt) - pd.Timedelta(days=interval_days) - pd.Timedelta(days=20)
        self.end_date = pd.to_datetime(running_dt)

    def calculate_check_out_ratio(self):
        """
        计算月份特征
        """
        # 确保日期列是 datetime 类型
        self.index_data['stats_dt'] = pd.to_datetime(self.index_data['stats_dt'])
        self.check_data['receive_dt'] = pd.to_datetime(self.check_data['receive_dt'])

        self.check_data = self.check_data.rename(columns={'receive_dt': 'stats_dt'})
        self.check_data = self.check_data.rename(columns={'org_inv_dk': 'pigfarm_dk'})
        # 筛选近400天的数据

        self.check_data = self.check_data[
            (self.check_data['stats_dt'] >= self.start_date) & 
            (self.check_data['stats_dt'] <= self.end_date)
        ]

        self.check_data = self.pick_prrs_data(self.check_data)

        self.calculate_prrs_check_out_ratio()
        
        return self.index_data
    
    def pick_prrs_data(self, check_data):
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
        final_data.to_csv(DataPathConfig.PRRS_PICK_DATA_SAVE_PATH, index=False, encoding='utf-8')
        
        logger.info(f"筛选前数据量: {len(check_data)}, 筛选后数据量: {len(final_data)}")
        
        return final_data
    
    def calculate_prrs_check_out_ratio(self):
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
        if len(self.check_data) == 0:
            logger.warning("没有检测数据可供处理")
            return self.index_data
        
        # 为check_data添加标记，标识是否为野毒数据
        # 条件1: check_item_dk在wild_check_items中
        # 条件2: check_item_dk在special_items中且index_item_dk在wild_indexes中
        self.check_data['is_wild'] = (
            self.check_data['check_item_dk'].isin(wild_check_items) | 
            ((self.check_data['check_item_dk'].isin(special_items)) & 
             (self.check_data['index_item_dk'].isin(wild_indexes)))
        )
        
        # 分别计算每个猪场每天的检测总量和阳性数量
        # 使用groupby优化计算效率
        result_data = []
        
        # 先获取所有唯一的日期和猪场组合
        unique_dates = self.index_data['stats_dt'].unique()
        unique_farms = self.index_data['pigfarm_dk'].unique()
        
        # 预处理数据，提高后续计算效率
        check_data_preprocessed = self.check_data[['pigfarm_dk', 'stats_dt', 'check_qty', 'check_out_qty', 'is_wild']].copy()
        
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
            self.index_data = self.index_data.merge(
                result_df, 
                on=['pigfarm_dk', 'stats_dt'], 
                how='left'
            )
            
            # 填充空值
            self.index_data['check_out_ratio_7d'] = self.index_data['prrs_7d_positive_rate']
            self.index_data['wild_check_out_ratio_7d'] = self.index_data['prrs_wild_7d_positive_rate']
        else:
            # 如果没有结果数据，添加默认列
            self.index_data['check_out_ratio_7d'] = np.nan
            self.index_data['wild_check_out_ratio_7d'] = np.nan
        
        logger.info(f"计算完成PRRS 7天阳性率和野毒阳性率，总数据量: {len(self.index_data)}")
        
        return self.index_data