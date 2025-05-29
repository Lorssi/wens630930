import pandas as pd
from tqdm import tqdm
from configs.feature_config import ColumnsConfig,DataPathConfig
import numpy as np
# class SurroundingPreprocessing:
#     def __init__(self, index_data=None):
#         self.index_data = index_data.copy()
#         self.calculate_surrounding_data = pd.read_csv(
#             DataPathConfig.ML_AOBRTION_RATE_DATA_SAVE_PATH,
#         )
#         self.org_data = pd.read_csv(
#             DataPathConfig.DIM_ORG_INV_PATH,
#         )
            

#     def calculate_surrounding_feature(self):
#         # 确保stats_dt是datetime类型
#         self.index_data['stats_dt'] = pd.to_datetime(self.index_data['stats_dt'])
        
#         # 1. 计算同三级公司同天的平均流产率
#         same_day_avg = self.index_data.groupby(['l3_org_inv_dk', 'stats_dt'])['abortion_rate'].mean().reset_index()
#         same_day_avg = same_day_avg.rename(columns={'abortion_rate': 'l3_abortion_mean'})
        
#         # 合并结果
#         # same_day_avg['stats_dt'] = same_day_avg['stats_dt'] + pd.Timedelta(days=1)  # 模拟当天没有数据
#         self.index_data = self.index_data.merge(same_day_avg, on=['l3_org_inv_dk', 'stats_dt'], how='left')
        
#         # 2. 使用pandas rolling功能优化滚动平均值计算
#         # 按l3_org_inv_dk分组并按日期排序
#         self.index_data = self.index_data.sort_values(['l3_org_inv_dk', 'stats_dt'])
        
#         # 创建每日汇总数据用于rolling计算
#         daily_data = self.index_data.groupby(['l3_org_inv_dk', 'stats_dt'])['abortion_rate'].mean().reset_index()
        
#         # 按组织分组计算rolling average
#         rolling_results = []
        
#         for org_id, group_data in tqdm(daily_data.groupby('l3_org_inv_dk'), desc="计算滚动平均"):
#             group_data = group_data.sort_values('stats_dt').set_index('stats_dt')
            
#             # 使用rolling函数计算不同窗口的平均值
#             group_data['l3_abortion_mean_7d'] = group_data['abortion_rate'].rolling(window='7D').mean()
#             group_data['l3_abortion_mean_15d'] = group_data['abortion_rate'].rolling(window='15D').mean()
#             group_data['l3_abortion_mean_30d'] = group_data['abortion_rate'].rolling(window='30D').mean()
            
#             group_data = group_data.reset_index()
#             group_data['l3_org_inv_dk'] = org_id
#             rolling_results.append(group_data[['l3_org_inv_dk', 'stats_dt', 'l3_abortion_mean_7d', 'l3_abortion_mean_15d', 'l3_abortion_mean_30d']])
        
#         # 合并所有结果
#         rolling_avg_df = pd.concat(rolling_results, ignore_index=True)
        
#         # 合并到原数据
#         # rolling_avg_df['stats_dt'] = rolling_avg_df['stats_dt'] + pd.Timedelta(days=1)  # 模拟当天没有数据
#         self.index_data = self.index_data.merge(
#             rolling_avg_df, 
#             on=['l3_org_inv_dk', 'stats_dt'], 
#             how='left'
#         )

#         return self.index_data

class SurroundingPreprocessing:
    def __init__(self, index_data=None):
        self.index_data = index_data.copy()
        self.calculate_surrounding_data = pd.read_csv(
            DataPathConfig.ML_AOBRTION_RATE_DATA_SAVE_PATH,
        )
        self.org_data = pd.read_csv(
            DataPathConfig.DIM_ORG_INV_PATH,
        )
            

    def calculate_surrounding_feature(self):
        # 准备数据
        self.calculate_surrounding_data = self.calculate_surrounding_data[['stats_dt', 'pigfarm_dk', 'abortion_rate']]
        # 确保stats_dt是datetime类型
        self.index_data['stats_dt'] = pd.to_datetime(self.index_data['stats_dt'])
        self.calculate_surrounding_data['stats_dt'] = pd.to_datetime(self.calculate_surrounding_data['stats_dt'])
        self.org_data.rename(columns={'org_inv_dk': 'pigfarm_dk'}, inplace=True)

        # 合并组织数据，获取每个猪场对应的三级公司
        self.calculate_surrounding_data = self.calculate_surrounding_data.merge(
            self.org_data[['pigfarm_dk', 'l3_org_inv_dk']],
            on='pigfarm_dk',
            how='left'
        )
        
        # 1. 计算同三级公司同天的平均流产率
        l3_same_day_avg = self.calculate_surrounding_data.groupby(['l3_org_inv_dk', 'stats_dt'])['abortion_rate'].mean().reset_index()
        l3_same_day_avg.rename(columns={'abortion_rate': 'l3_abortion_mean'}, inplace=True)
        
        # 2. 计算同三级公司近7天的平均流产率（使用高效的方式）
        # 按三级公司和日期排序
        self.calculate_surrounding_data = self.calculate_surrounding_data.sort_values(['l3_org_inv_dk', 'stats_dt'])
        
        # 按三级公司分组计算7天窗口均值
        rolling_results = []
        
        for l3_id, group_data in tqdm(self.calculate_surrounding_data.groupby('l3_org_inv_dk'), desc="计算滚动平均"):
            # 按日期分组，计算每天的平均值（避免同一天多个猪场重复计算）
            daily_avg = group_data.groupby('stats_dt')['abortion_rate'].mean().reset_index()
            daily_avg = daily_avg.sort_values('stats_dt').set_index('stats_dt')
            
            # 使用rolling函数计算7天窗口的平均值
            daily_avg['l3_abortion_mean_7d'] = daily_avg['abortion_rate'].rolling(window='7D').mean()
            daily_avg['l3_abortion_mean_15d'] = daily_avg['abortion_rate'].rolling(window='15D').mean()
            daily_avg['l3_abortion_mean_30d'] = daily_avg['abortion_rate'].rolling(window='30D').mean()
            
            daily_avg = daily_avg.reset_index()
            daily_avg['l3_org_inv_dk'] = l3_id
            rolling_results.append(daily_avg[['l3_org_inv_dk', 'stats_dt', 'l3_abortion_mean_7d','l3_abortion_mean_15d', 'l3_abortion_mean_30d']])
        
        # 合并所有结果
        rolling_avg_df = pd.concat(rolling_results, ignore_index=True)
        
        # 合并结果到原数据
        self.index_data = self.index_data.merge(
            l3_same_day_avg,
            on=['l3_org_inv_dk', 'stats_dt'],
            how='left'
        )
        
        self.index_data = self.index_data.merge(
            rolling_avg_df[['l3_org_inv_dk', 'stats_dt', 'l3_abortion_mean_7d','l3_abortion_mean_15d', 'l3_abortion_mean_30d']],
            on=['l3_org_inv_dk', 'stats_dt'],
            how='left'
        )
        
        return self.index_data