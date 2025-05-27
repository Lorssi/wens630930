import pandas as pd
import numpy as np
from utils.logger import setup_logger
from configs.logger_config import logger_config
from tqdm import tqdm
logger = setup_logger(logger_config.TRAIN_LOG_FILE_PATH, logger_name="PreprocessingLogger")

import logging
logger = logging.getLogger(__name__)

class IntroDataPreprocessor:
    def __init__(self, intro_data_path, tame_data_path, index_data):
        self.intro_data_path = intro_data_path
        self.tame_data_path = tame_data_path
        # self.index_data = index_data
        self.feature_columns = ['intro_source_num_90d', 'intro_source_is_single', 'intro_times_30d', 'intro_times_90d']
        self.load_data()

    def load_data(self):
        # 加载引种数据
        try:
            self.intro_data = pd.read_csv(self.intro_data_path, encoding='utf-8')
            # 把来源猪场的空值由供应商名字填充
            self.intro_data['cffromhogp_nm'] = self.intro_data['cffromhogp_nm'].fillna(self.intro_data['vendor_nm'])
            # 排除 boar_src_type 为 '选留' 的记录
            self.intro_data = self.intro_data[self.intro_data['boar_src_type'] != '选留']
            self.intro_data['intro_dt'] = pd.to_datetime(self.intro_data['intro_dt'])
        except Exception as e:
            logger.error(f"加载引种数据失败: {e}")
            self.intro_data = pd.DataFrame()  # 失败时初始化为空数据框

        # 加载入群数据
        try:
            self.tame_data = pd.read_csv(self.tame_data_path, encoding='utf-8')
        except Exception as e:
            logger.error(f"加载驯化数据失败: {e}")
            self.tame_data = pd.DataFrame()  # 失败时初始化为空数据框

    def calculate_intro_feature(self):
        """
        计算引种特征 - 高效优化版，先预处理数据再计算特征
        """
        if self.intro_data.empty:
            logger.warning("引种数据为空，无法计算特征")
            return self.index_data
            
        # 避免在原始数据上修改
        result = self.index_data.copy()
        
        # 转换日期列
        result['stats_dt'] = pd.to_datetime(result['stats_dt'])
        
        logger.info("预处理数据计算特征...")
        
        # 添加性能优化: 为intro_data添加索引
        self.intro_data.sort_values(['org_inv_dk', 'intro_dt'], inplace=True)
        
        # 获取所有需要计算特征的日期
        all_dates = result['stats_dt'].unique()
        min_date = result['stats_dt'].min()
        max_date = result['stats_dt'].max()
        
        logger.info(f"创建所有猪场的特征映射表 ({min_date} 到 {max_date})...")
        
        # 创建一个空DataFrame用于存储所有特征
        farm_ids = result['pigfarm_dk'].unique()
        
        # 初始化一个字典存储所有特征
        farm_features = {}
        
        # 为每个猪场预计算特征
        for farm_id in tqdm(farm_ids, desc="计算猪场特征"):

            # 筛选该猪场的所有引种记录
            farm_intro = self.intro_data[self.intro_data['org_inv_dk'] == farm_id]

            if farm_intro.empty:
                continue
                
            # 为这个猪场创建一个日期字典
            farm_dict = {}
                
            # 对每个日期计算特征 (只计算index_data中的日期)
            for date in all_dates:
                # 计算90天内的特征
                mask_90d = (farm_intro['intro_dt'] <= date) & (farm_intro['intro_dt'] >= date - pd.Timedelta(days=90))
                intro_90d = farm_intro[mask_90d]
                
                # 计算30天内的特征
                mask_30d = (farm_intro['intro_dt'] <= date) & (farm_intro['intro_dt'] >= date - pd.Timedelta(days=30))
                intro_30d = farm_intro[mask_30d]

                # 计算特征值
                if len(intro_90d) > 0:
                    unique_sources = intro_90d['cffromhogp_nm'].nunique()
                    farm_dict[date] = {
                        'intro_source_num_90d': unique_sources,
                        'intro_source_is_single': 1 if unique_sources <= 1 else 0,
                        'intro_times_90d': len(intro_90d),
                        'intro_times_30d': len(intro_30d)
                    }
                else:
                    farm_dict[date] = {
                        'intro_source_num_90d': 0,
                        'intro_source_is_single': 0, 
                        'intro_times_90d': 0,
                        'intro_times_30d': 0
                    }
                    
            # 存储这个猪场的所有特征
            farm_features[farm_id] = farm_dict
        
        logger.info("将预计算特征合并到结果中...")
        
        # 将特征合并到结果DataFrame中，模拟T+1架构
        new_features = []
        for _, row in tqdm(result.iterrows(), total=len(result), desc="合并特征"):
            farm_id = row['pigfarm_dk']
            stats_date = row['stats_dt']
            
            # 查找前一天的特征，模拟T+1
            prev_date = stats_date - pd.Timedelta(days=1)
            
            if farm_id in farm_features and prev_date in farm_features[farm_id]:
                feature_dict = {'pigfarm_dk': farm_id, 'stats_dt': stats_date}
                feature_dict.update(farm_features[farm_id][prev_date])
                new_features.append(feature_dict)
        
        # 将特征转换为DataFrame并与结果合并
        if new_features:
            features_df = pd.DataFrame(new_features)
            result = pd.merge(result, features_df, on=['pigfarm_dk', 'stats_dt'], how='left')
        
        # 填充缺失的特征值为0
        for col in self.feature_columns:
            if col in result.columns:
                result[col] = result[col].fillna(0)
            else:
                result[col] = 0
        
        logger.info("引种特征计算完成")

        return result