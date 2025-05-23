import pandas as pd
import numpy as np
from utils.logger import setup_logger
from configs.logger_config import logger_config
from tqdm import tqdm
logger = setup_logger(logger_config.TRAIN_LOG_FILE_PATH, logger_name="PreprocessingLogger")

class IntroDataPreprocessor:
    def __init__(self, data_path, index_data, running_dt= "2024-10-1", interval_days=400):
        self.data_path = data_path
        self.id_column = 'org_inv_dk'
        self.date_column = 'intro_dt' 
        self.vendor_nm = 'vendor_nm' # 供应商名称
        self.cffromhogp_nm = 'cffromhogp_nm' # 来源猪场名称

        self.index_data = index_data.copy()  # 深拷贝，避免对原数据的修改
        self.index_id_column = 'pigfarm_dk'
        self.index_date_column = 'stats_dt'

        self.end_date = pd.to_datetime(running_dt) - pd.Timedelta(days=1)  # 计算截止日期
        self.intro_start_date = self.end_date - pd.Timedelta(days=interval_days) - pd.Timedelta(days=95)
        self.intro_data = self.load_data()

        if self.intro_data is not None:
            # 确保在计算前，数据是按猪场和日期排序的
            self.intro_data.sort_values(by=[self.date_column, self.id_column], inplace=True)
            logger.info(f"引种数据按 '{self.id_column}' 和 '{self.date_column}' 排序完成")

    def load_data(self):
        """加载数据"""
        try:
            df = pd.read_csv(self.data_path, encoding='utf-8')
            logger.info(f"成功加载引种数据: {self.data_path}")

            df[self.date_column] = pd.to_datetime(df[self.date_column])  # 转换为 datetime 格式

            # 过滤数据，确保日期在指定范围内
            df = df[(df[self.date_column] >= self.intro_start_date) & (df[self.date_column] <= self.end_date)]
            # 排序移到 __init__ 中，在加载后统一执行，确保后续操作基于正确的顺序
            return df
        except FileNotFoundError:
            logger.error(f"错误: 数据文件未找到于 {self.data_path}")
            return None
        except KeyError as e:
            logger.error(f"加载数据时发生列名错误: {e}. 请检查CSV文件中的列名是否与期望的列名匹配 ({self.date_column}, {self.id_column} 等).")
            return None
        
    def calculate_is_single_and_intro_num(self):
        self.intro_data['intro_source'] = self.intro_data[self.cffromhogp_nm].fillna(self.intro_data[self.vendor_nm])

        index_group = self.index_data.groupby(self.index_id_column)
        intro_group = self.intro_data.groupby(self.id_column)

        for index_id, index_group_df in tqdm(index_group, desc="Processing Index Data", total=len(index_group)):
            # 获取当前猪场的引种数据
            intro_data = intro_group.get_group(index_id) if index_id in intro_group.groups else None

            if intro_data is not None:
                for index, row in index_group_df.iterrows():
                    # 获取当前猪场的日期
                    current_date = row[self.index_date_column]
                    # 计算当前日期前7天的日期范围
                    start_date = current_date - pd.Timedelta(days=89)
                    end_date = current_date

                    # 筛选出在当前日期前7天内的引种数据
                    intro_data_filtered = intro_data[(intro_data[self.date_column] >= start_date) & 
                                                     (intro_data[self.date_column] <= end_date)]

                    # 计算 is_single 和 intro_num
                    if not intro_data_filtered.empty:
                        intro_num = intro_data_filtered['intro_dt'].nunique()  # 引种数量
                        is_single = 0 if intro_data_filtered['intro_source'].nunique() > 1 else 1
                        
                        self.index_data.loc[index, 'is_single'] = is_single
                        self.index_data.loc[index, 'intro_num'] = intro_num

        return self.index_data