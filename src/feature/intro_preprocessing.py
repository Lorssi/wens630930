import pandas as pd
import numpy as np
from utils.logger import setup_logger
from configs.logger_config import logger_config
from tqdm import tqdm
logger = setup_logger(logger_config.TRAIN_LOG_FILE_PATH, logger_name="PreprocessingLogger")

class IntroDataPreprocessor:
    def __init__(self, intro_data_path, tame_data_path, index_data, running_dt=None, interval_days=None):
        self.intro_data_path = intro_data_path
        self.tame_data_path = tame_data_path
        self.index_data = index_data
        self.feature_columns = ['intro_source_num_90d', 'intro_source_is_single', 'intro_times_30d', 'intro_times_90d']
        self.end_dt = pd.to_datetime(running_dt) - pd.Timedelta(days=1)
        self.start_dt = pd.to_datetime(running_dt) - pd.Timedelta(days=interval_days)
        self.extend_dt = self.start_dt - pd.Timedelta(days=91) # 扩展到90天前，用于计算90天内的引种特征

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
            # 过滤日期范围，减少数据大小
            self.intro_data = self.intro_data[(self.intro_data['intro_dt'] >= self.extend_dt) & (self.intro_data['intro_dt'] <= self.end_dt)]
            # 排序，用于加快计算速度
            self.intro_data.sort_values(['org_inv_dk', 'intro_dt'], inplace=True)
        except Exception as e:
            logger.error(f"加载引种数据失败: {e}")
            self.intro_data = pd.DataFrame()  # 失败时初始化为空数据框

        # 加载入群数据
        try:
            self.tame_data = pd.read_csv(self.tame_data_path, encoding='utf-8', low_memory=False)
            self.tame_data.rename(columns={'tmp_ads_pig_isolation_tame_risk_l1_n2.org_inv_dk': 'org_inv_dk', 'tmp_ads_pig_isolation_tame_risk_l1_n2.bill_dt': 'bill_dt'}, inplace=True)
            self.tame_data['bill_dt'] = pd.to_datetime(self.tame_data['bill_dt'])
            # 过滤日期范围，减少数据大小
            self.tame_data = self.tame_data[(self.tame_data['bill_dt'] >= self.extend_dt) & (self.tame_data['bill_dt'] <= self.end_dt)]
            # 排序，用于加快计算速度
            self.tame_data.sort_values(['org_inv_dk', 'bill_dt'], inplace=True)
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
        index_data_copy = self.index_data.copy()
        
        logger.info("预处理数据计算特征...")
                
 
        # 获取所有需要计算特征的日期（T+1模式）
        all_stats_dates = sorted(index_data_copy['stats_dt'].unique())
        # 获取需要计算的日期（T+1模式下是stats_dt日期的前一天）
        all_calc_dates = [date - pd.Timedelta(days=1) for date in all_stats_dates]
        
        # 创建一个空DataFrame用于存储所有特征
        farm_ids = index_data_copy['pigfarm_dk'].unique()
        
        # 初始化一个字典存储所有特征
        farm_features = {}
        
        # 为每个猪场预计算特征
        for farm_id in tqdm(farm_ids, desc="计算猪场特征"):
            # 筛选该猪场的所有引种记录
            farm_intro = self.intro_data[self.intro_data['org_inv_dk'] == farm_id]
            # 筛选该猪场的所有入群记录
            farm_tame = self.tame_data[self.tame_data['org_inv_dk'] == farm_id]
            
            has_intro = not farm_intro.empty
            has_tame = not farm_tame.empty
            
            if not has_intro and not has_tame:
                continue
                
            # 为这个猪场创建一个日期字典
            farm_dict = {}
                
            # 对每个日期计算特征 (只计算index_data中的日期)
            for date in all_calc_dates:
                feature_dict = {
                    'intro_source_num_90d': 0,
                    'intro_source_is_single': 0, 
                    'intro_times_90d': 0,
                    'intro_times_30d': 0,
                    'boar_tame_num_30d': 0
                }
                
                # 计算引种特征
                if has_intro:
                    # 创建90天和30天的过滤条件
                    mask_90d = (farm_intro['intro_dt'] <= date) & (farm_intro['intro_dt'] >= date - pd.Timedelta(days=90))
                    mask_30d = (farm_intro['intro_dt'] <= date) & (farm_intro['intro_dt'] >= date - pd.Timedelta(days=30))
                    
                    # 直接计算特征值，无需创建中间DataFrame
                    if mask_90d.any():  # 如果有任何符合条件的行
                        # 计算90天内的唯一来源数
                        unique_sources = farm_intro.loc[mask_90d, 'cffromhogp_nm'].nunique()
                        
                        # 更新特征值
                        feature_dict.update({
                            'intro_source_num_90d': unique_sources,
                            'intro_source_is_single': 1 if unique_sources <= 1 else 0,
                            'intro_times_90d': mask_90d.sum(),  # 直接对掩码求和获取行数
                            'intro_times_30d': mask_30d.sum()   # 直接对掩码求和获取行数
                        })
                
                # 计算入群特征
                if has_tame:
                    # 直接计算30天内入群数量，无需创建中间DataFrame
                    mask_tame_30d = (farm_tame['bill_dt'] <= date) & (farm_tame['bill_dt'] >= date - pd.Timedelta(days=30))
                    feature_dict['boar_tame_num_30d'] = mask_tame_30d.sum()
                    
                farm_dict[date] = feature_dict
                        
            # 存储这个猪场的所有特征
            farm_features[farm_id] = farm_dict
        
        logger.info("将预计算特征合并到结果中...")
        
        # 将特征合并到结果DataFrame中，模拟T+1架构
        new_features = []
        for _, row in tqdm(index_data_copy.iterrows(), total=len(index_data_copy), desc="合并特征"):
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
            index_data_copy = pd.merge(index_data_copy, features_df, on=['pigfarm_dk', 'stats_dt'], how='left')
        
        logger.info("引种特征计算完成")

        return index_data_copy