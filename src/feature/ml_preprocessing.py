# data_loader/preprocessing.py
import pandas as pd
import numpy as np
from utils.logger import setup_logger
from configs.logger_config import logger_config
from configs.feature_config import DataPathConfig, ColumnsConfig
logger = setup_logger(logger_config.TRAIN_LOG_FILE_PATH, logger_name="PreprocessingLogger")

class MLDataPreprocessor:
    def __init__(self, data_path, running_dt= "2024-10-1", interval_days=400):
        self.data_path = data_path
        self.id_column = 'pigfarm_dk'
        self.date_column = 'stats_dt'
        self.abort_qty_column = 'abort_qty'
        self.preg_stock_qty_column = 'preg_stock_qty'
        
        self.index_end_date = pd.to_datetime(running_dt) - pd.Timedelta(days=1)  # 计算截止日期
        self.index_start_date = self.index_end_date - pd.Timedelta(days=interval_days)
        self.ml_end_date = pd.to_datetime(running_dt) - pd.Timedelta(days=1)  # 计算截止日期
        self.ml_start_date = pd.to_datetime(running_dt) - pd.Timedelta(days=interval_days) - pd.Timedelta(days=10)

        self.index_data = self.load_data(mode='index')
        self.ml_data = self.load_data(mode='ml')


    def load_data(self, mode='ml'):
        """加载数据"""
        try:
            df = pd.read_csv(self.data_path, encoding='utf-8')
            logger.info(f"成功加载数据: {self.data_path}")

            # 注意：你原来的去重键是 'stats_dt, pigfarm_dk' (一个字符串)
            # 应该是一个列名列表 ['stats_dt', 'pigfarm_dk']
            # 我假设你的意图是针对猪场ID和日期这两个列去重
            duplicate_subset = [self.date_column, self.id_column]
            df.drop_duplicates(subset=duplicate_subset, inplace=True)
            logger.info(f"ML数据去重完成, 去重subset={duplicate_subset}")

            df[self.date_column] = pd.to_datetime(df[self.date_column])  # 转换为 datetime 格式

            # 过滤数据，确保日期在指定范围内
            if mode == 'index':
                df = df[(df[self.date_column] >= self.index_start_date) & (df[self.date_column] <= self.index_end_date)]
            elif mode == 'ml':
                df = df[(df[self.date_column] >= self.ml_start_date) & (df[self.date_column] <= self.ml_end_date)]

            # 处理数据
            if df is not None:
                # 确保在计算前，数据是按猪场和日期排序的
                df.sort_values(by=[self.date_column, self.id_column], inplace=True)
                logger.info(f"ML数据按 '{self.id_column}' 和 '{self.date_column}' 排序完成")

            return df
        except FileNotFoundError:
            logger.error(f"错误: 数据文件未找到于 {self.data_path}")
            return None
        except KeyError as e:
            logger.error(f"加载数据时发生列名错误: {e}. 请检查CSV文件中的列名是否与期望的列名匹配 ({self.date_column}, {self.id_column} 等).")
            return None

    def calculate_abortion_rate(self, new_column_name='abortion_rate'):
        """
        计算流产率。
        流产率 = sum(近7天流产数量) / (sum(近7天流产数量) + 当天怀孕母猪存栏量)
        """
        if self.ml_data is None:
            logger.error("数据未加载，无法计算流产率")
            return

        if self.abort_qty_column not in self.ml_data.columns:
            logger.error(f"错误: 流产数量列 '{self.abort_qty_column}' 在数据中未找到。")
            return
        if self.preg_stock_qty_column not in self.ml_data.columns:
            logger.error(f"错误: 怀孕母猪存栏量列 '{self.preg_stock_qty_column}' 在数据中未找到。")
            return

        logger.info(f"开始计算流产率，存入新列 '{new_column_name}'...")

        # self.ml_data[self.date_column] = self.ml_data[self.date_column] + pd.DateOffset(days=1)  # 将日期加1天，确保预警运行边界日期的特征为观察期窗口特征

        # 确保流产数量和怀孕母猪存栏量是数值类型，并将NaN填充为0，因为它们参与计算
        self.ml_data[self.abort_qty_column] = pd.to_numeric(self.ml_data[self.abort_qty_column], errors='coerce').fillna(0)
        self.ml_data[self.preg_stock_qty_column] = pd.to_numeric(self.ml_data[self.preg_stock_qty_column], errors='coerce').fillna(0)
        
        # 使用 groupby 和 rolling window 计算每个猪场每个日期的近7天流产总数
        # 'closed="left"' 通常用于rolling sum，但这里我们需要包含当天，所以默认'right'就可以
        # min_periods=1 表示即使不足7天，也会计算已有的天数和
        self.ml_data['recent_7day_abort_sum'] = self.ml_data.groupby(self.id_column)[self.abort_qty_column]\
                                                            .rolling(window=7, min_periods=7).sum()\
                                                            .reset_index(level=0, drop=True) # reset_index 去掉 groupby 带来的多级索引

        # 定义一个函数来计算流产率，处理分母为0的情况
        def calculate_rate(row):
            sum_recent_abort = row['recent_7day_abort_sum']
            current_preg_stock = row[self.preg_stock_qty_column]

            # 如果7天流产总和是NaN (因为窗口不足7天)，则流产率也是NaN
            if pd.isna(sum_recent_abort):
                return np.nan
            
            # 如果怀孕母猪存栏量是NaN，则流产率也是NaN
            if pd.isna(current_preg_stock):
                return np.nan
            
            numerator = sum_recent_abort
            denominator = sum_recent_abort + current_preg_stock
            
            if denominator == 0:
                return np.nan  # 或者 np.nan，根据业务需求决定如何处理分母为0的情况
            else:
                return numerator / denominator

        # 应用计算函数
        self.ml_data[new_column_name] = self.ml_data.apply(calculate_rate, axis=1)

        self.index_data = pd.merge(self.index_data, self.ml_data[[self.date_column, self.id_column, new_column_name]],
                                   on=[self.date_column, self.id_column], how='left')
        
        # 可以选择删除中间列
        # self.ml_data.drop(columns=['recent_7day_abort_sum'], inplace=True)

        logger.info(f"成功计算流产率，并存入列 '{new_column_name}'。")

    def clean_data(self):
        """
        删除流产率全为空的猪场数据
        如果某个猪场的所有记录中abortion_rate均为NaN，则删除该猪场的所有记录
        """
        logger.info("开始将流产率全空的猪场删除...")
        
        if self.index_data is None or 'abortion_rate' not in self.index_data.columns:
            logger.error("无法清理数据：index_data为空或没有abortion_rate列")
            return
        
        # 记录清理前的数据状态
        before_count = len(self.index_data)
        before_farm_count = self.index_data[self.id_column].nunique()
        
        # 识别流产率全为空的猪场
        # 对每个猪场，检查其abortion_rate是否全为NaN
        farm_abortion_status = self.index_data.groupby(self.id_column)['abortion_rate'].apply(
            lambda x: not x.isna().all()  # 返回True表示至少有一个非空值
        )
        
        # 获取要保留的猪场ID列表（至少有一个非空流产率）
        farms_to_keep = farm_abortion_status[farm_abortion_status].index.tolist()
        
        # 只保留这些猪场的数据
        self.index_data = self.index_data[self.index_data[self.id_column].isin(farms_to_keep)]
        
        # 计算清理结果
        after_count = len(self.index_data)
        after_farm_count = self.index_data[self.id_column].nunique()
        removed_records = before_count - after_count
        removed_farms = before_farm_count - after_farm_count
        
        logger.info(f"数据清理完成：删除了{removed_farms}个流产率全为空的猪场，共{removed_records}条记录")
        logger.info(f"清理后数据：{after_count}条记录，{after_farm_count}个猪场")

        self.index_data.to_csv(DataPathConfig.INDEX_ABORTION_RATE_DATA_SAVE_PATH, index=False, encoding='utf-8-sig')
        logger.info(f"INDEX流产率数据已保存至 {DataPathConfig.INDEX_ABORTION_RATE_DATA_SAVE_PATH}")

    def clean_ml_data(self):
        """
        删除流产率全为空的猪场数据
        如果某个猪场的所有记录中abortion_rate均为NaN，则删除该猪场的所有记录
        """
        logger.info("开始将流产率全空的猪场删除...")
        
        if self.ml_data is None or 'abortion_rate' not in self.ml_data.columns:
            logger.error("无法清理数据：index_data为空或没有abortion_rate列")
            return
        
        # 记录清理前的数据状态
        before_count = len(self.ml_data)
        before_farm_count = self.ml_data[self.id_column].nunique()
        
        # 识别流产率全为空的猪场
        # 对每个猪场，检查其abortion_rate是否全为NaN
        farm_abortion_status = self.ml_data.groupby(self.id_column)['abortion_rate'].apply(
            lambda x: not x.isna().all()  # 返回True表示至少有一个非空值
        )
        
        # 获取要保留的猪场ID列表（至少有一个非空流产率）
        farms_to_keep = farm_abortion_status[farm_abortion_status].index.tolist()
        
        # 只保留这些猪场的数据
        self.ml_data = self.ml_data[self.ml_data[self.id_column].isin(farms_to_keep)]
        
        # 计算清理结果
        after_count = len(self.ml_data)
        after_farm_count = self.ml_data[self.id_column].nunique()
        removed_records = before_count - after_count
        removed_farms = before_farm_count - after_farm_count
        
        logger.info(f"数据清理完成：删除了{removed_farms}个流产率全为空的猪场，共{removed_records}条记录")
        logger.info(f"清理后数据：{after_count}条记录，{after_farm_count}个猪场")

        self.ml_data.to_csv(DataPathConfig.ML_AOBRTION_RATE_DATA_SAVE_PATH, index=False, encoding='utf-8-sig')
        logger.info(f"ML流产率计算完成，数据已保存至 {DataPathConfig.ML_AOBRTION_RATE_DATA_SAVE_PATH}")