import pandas as pd
from configs.feature_config import DataPathConfig,ColumnsConfig
from configs.logger_config import logger_config
from tqdm import tqdm
import numpy as np

from utils.logger import setup_logger

logger = setup_logger(logger_config.TRAIN_LOG_FILE_PATH, logger_name="LabelLogger")

class LabelGenerator:
    def __init__(self, feature_data, running_dt="2024-10-01", interval_days=400):
        self.index_data = feature_data.copy()  # 深拷贝，避免对原数据的修改

        self.id_column = 'pigfarm_dk'
        self.date_column = 'stats_dt'
        self.calculate_data = self.load_data()

        self.running_dt = pd.to_datetime(running_dt)
        self.interval_days = interval_days

    def load_data(self):
        """加载数据"""
        try:
            df = pd.read_csv(DataPathConfig.ML_AOBRTION_RATE_DATA_SAVE_PATH, encoding='utf-8')
            logger.info(f"成功加载数据: {DataPathConfig.ML_AOBRTION_RATE_DATA_SAVE_PATH}")

            df[self.date_column] = pd.to_datetime(df[self.date_column])  # 转换为 datetime 格式

            # 处理数据
            if df is not None:
                # 确保在计算前，数据是按猪场和日期排序的
                df.sort_values(by=[self.date_column, self.id_column], inplace=True)

            return df
        except FileNotFoundError:
            logger.error(f"错误: 数据文件未找到于 {DataPathConfig.ML_AOBRTION_RATE_DATA_SAVE_PATH}")
            return None
        
    def has_risk_4_class_period_generate_label(self):
        """
        生成标签：如果未来7天内流产率超过0.0025，则标记为1，否则为0
        
        Returns:
            DataFrame: 包含标签列的特征数据
        """
        label = ColumnsConfig.HAS_RISK_LABEL
        periods = [(1,7),(8,14),(15,21)]

        days_label_list = [ColumnsConfig.DAYS_RISK_8_CLASS_PRE.format(period[0], period[1]) for period in periods]

        if self.calculate_data is None or len(self.calculate_data) == 0:
            logger.info("Label计算数据为空，无法生成标签")
            return None
        if self.index_data is None or len(self.index_data) == 0:
            logger.info("INDEX数据为空，无法生成标签")
            return None
        
        # 确保日期列是datetime类型
        self.calculate_data[self.date_column] = pd.to_datetime(self.calculate_data[self.date_column])
        self.index_data[self.date_column] = pd.to_datetime(self.index_data[self.date_column])
        
        # 为数据添加标签列，默认为0（无风险）
        # self.feature_data['label'] = 0
        
        # 将数据按猪场ID分组
        grouped_data = self.calculate_data.groupby(self.id_column)
        
        # 对每个猪场处理
        for farm_id, farm_data in tqdm(grouped_data):
            # 确保数据按日期排序
            farm_data = farm_data.sort_values(by=self.date_column)
            
            # 对每个日期记录检查未来7天的风险
            for index, row in farm_data.iterrows():
                current_date = row[self.date_column]
                
                for period in periods:
                    left, right = period[0], period[1] # 1-7天，8-14天，15-21天
                    pre_label = ColumnsConfig.HAS_RISK_4_CLASS_PRE.format(left, right)
                    days_label = ColumnsConfig.DAYS_RISK_8_CLASS_PRE.format(left, right)
                    # 计算未来7天的日期范围
                    future_start= current_date + pd.Timedelta(days=left)  # 从下一天开始
                    future_end = current_date + pd.Timedelta(days=right)    # 到7天后结束
                    
                    # 筛选未来7天的数据
                    future_data = farm_data[farm_data[self.date_column].between(future_start, future_end)]
                    
                    # 检查未来7天内是否有流产率超过阈值
                    if not future_data.empty:
                        # 计算有无标签
                        if (future_data['abortion_rate'] >= 0.0025).any():
                            self.calculate_data.loc[index, pre_label] = 1
                        else:
                            self.calculate_data.loc[index, pre_label] = 0
                        # 计算天数标签
                        risk_days_count = (future_data['abortion_rate'] >= 0.0025).sum()
                        self.calculate_data.loc[index, days_label] = risk_days_count


        def make_label(row):
            make_labels = [ColumnsConfig.HAS_RISK_4_CLASS_PRE.format(period[0], period[1]) for period in periods]
            if row[make_labels[0]] == 1:
                return 1
            elif row[make_labels[1]] == 1:
                return 2
            elif row[make_labels[2]] == 1:
                return 3
            else:
                return 0

        self.calculate_data[label] = self.calculate_data.apply(make_label, axis=1)
        self.index_data = pd.merge(self.index_data, self.calculate_data[[self.date_column, self.id_column, label] + days_label_list], on=[self.date_column, self.id_column], how='left')
        self.calculate_data.to_csv(DataPathConfig.ABORTION_CALCULATE_DATA_SAVE_PATH, index=False, encoding='utf-8-sig')
        self.index_data.to_csv(DataPathConfig.NON_DROP_NAN_LABEL_DATA_SAVE_PATH, index=False, encoding='utf-8-sig')
        
        # 统计标签分布
        total_records = len(self.index_data)
        # 计算非零标签的数量（即风险记录总数 - 类别1+2+3）
        risk_records = len(self.index_data[self.index_data[label] > 0])
        risk_percentage = (risk_records / total_records) * 100 if total_records > 0 else 0

        # 分别统计各类别的记录数和占比
        label_counts = self.index_data[label].value_counts().sort_index()
        logger.info(f"标签分布：")
        for lbl, count in label_counts.items():
            percentage = (count / total_records) * 100
            if lbl == 0:
                logger.info(f"  无风险 (类别 {lbl}): {count} 条记录 ({percentage:.2f}%)")
            else:
                risk_type = "近期(1-7天)" if lbl == 1 else "中期(8-14天)" if lbl == 2 else "远期(15-21天)"
                logger.info(f"  {risk_type} (类别 {lbl}): {count} 条记录 ({percentage:.2f}%)")

        logger.info(f"标签生成完成：总记录数 {total_records}，所有风险记录数 {risk_records} ({risk_percentage:.2f}%)")
        logger.info(f"去除无标签数据前，数据量为：{len(self.index_data)}")
        self.index_data.dropna(subset=[label], inplace=True)  # 删除没有标签的记录
        logger.info(f"删除无标签的记录后，数据量为: {len(self.index_data)}")

        # 保存带标签的数据
        if DataPathConfig.ABORTION_LABEL_DATA_SAVE_PATH:
            self.index_data.to_csv(DataPathConfig.ABORTION_LABEL_DATA_SAVE_PATH, index=False, encoding='utf-8')
            print(f"带标签的数据已保存至: {DataPathConfig.ABORTION_LABEL_DATA_SAVE_PATH}")

        X = self.index_data.drop(columns=[label] + days_label_list)
        y = self.index_data[['stats_dt', label] + days_label_list]

        return X, y

    def has_risk_4_class_period_generate_label_alter(self):
        """
        生成标签：如果未来7天内流产率超过0.0025，则标记为1，否则为0
        优化版本 - 修复索引问题和数据处理
        """
        label = ColumnsConfig.HAS_RISK_LABEL
        periods = [(1,7),(8,14),(15,21)]
        days_label_list = [ColumnsConfig.DAYS_RISK_8_CLASS_PRE.format(period[0], period[1]) for period in periods]

        if self.calculate_data is None or len(self.calculate_data) == 0:
            logger.info("Label计算数据为空，无法生成标签")
            return None
        if self.index_data is None or len(self.index_data) == 0:
            logger.info("INDEX数据为空，无法生成标签")
            return None
        
        # 确保日期列是datetime类型
        self.calculate_data[self.date_column] = pd.to_datetime(self.calculate_data[self.date_column])
        self.index_data[self.date_column] = pd.to_datetime(self.index_data[self.date_column])
        
        # 预初始化所有标签列
        for period in periods:
            left, right = period[0], period[1]
            pre_label = ColumnsConfig.HAS_RISK_4_CLASS_PRE.format(left, right)
            days_label = ColumnsConfig.DAYS_RISK_8_CLASS_PRE.format(left, right)
            self.calculate_data[pre_label] = np.nan
            self.calculate_data[days_label] = np.nan
        
        # 创建一个字典，存储每个猪场的数据，并保留原始索引
        farm_dict = dict(tuple(self.calculate_data.groupby(self.id_column)))
        total_count = len(self.calculate_data)
        processed_count = 0
        
        print("开始处理标签数据...")
        farm_count = 0
        # 处理每个猪场的数据
        for farm_id, farm_data in tqdm(farm_dict.items()):
            farm_count += 1
            # 创建猪场ID和日期到原始索引的映射
            date_to_idx = {}
            for idx, row in farm_data.iterrows():
                date_to_idx[row[self.date_column]] = idx
            
            # 按日期排序，但保持原始索引的引用
            farm_data_sorted = farm_data.sort_values(by=self.date_column)
            farm_dates = farm_data_sorted[self.date_column].values
            farm_rates = farm_data_sorted['abortion_rate'].values
            
            # 处理每个日期记录
            for i, (idx, row) in enumerate(farm_data_sorted.iterrows()):
                current_date = row[self.date_column]
                
                for period in periods:
                    left, right = period[0], period[1]
                    pre_label = ColumnsConfig.HAS_RISK_4_CLASS_PRE.format(left, right)
                    days_label = ColumnsConfig.DAYS_RISK_8_CLASS_PRE.format(left, right)
                    
                    # 计算未来日期范围
                    future_start = current_date + pd.Timedelta(days=left)
                    future_end = current_date + pd.Timedelta(days=right)
                    
                    # 首先检查结束日期是否存在于数据中（确保窗口完整）
                    end_date_exists = np.any(farm_dates == future_end)
                    
                    # 使用向量化操作找出符合日期范围的记录
                    future_mask = (farm_dates >= future_start) & (farm_dates <= future_end)
                    future_rates = farm_rates[future_mask]
                    
                    # 只有当结束日期存在且有足够的数据时才计算标签
                    if end_date_exists and len(future_rates) > 0:
                        # 计算风险标签
                        has_risk = np.any(future_rates >= 0.0025)
                        self.calculate_data.loc[idx, pre_label] = 1 if has_risk else 0
                        
                        # 计算风险天数
                        risk_days = np.sum(future_rates >= 0.0025)
                        self.calculate_data.loc[idx, days_label] = risk_days

        # 使用向量化操作生成最终标签
        def make_label_vec(df):
            make_labels = [ColumnsConfig.HAS_RISK_4_CLASS_PRE.format(period[0], period[1]) for period in periods]
            
            # 首先检查是否有任何NaN值
            has_nan_mask = df[make_labels].isna().any(axis=1)
            
            # 设置条件
            conditions = [
                (~has_nan_mask) & (df[make_labels[0]] == 1),
                (~has_nan_mask) & (df[make_labels[1]] == 1),
                (~has_nan_mask) & (df[make_labels[2]] == 1)
            ]
            choices = [1, 2, 3]
            
            # 创建结果数组，默认为0（无风险）
            result = np.select(conditions, choices, default=0)
            
            # 对有NaN的行，将结果设为NaN
            result = pd.Series(result, index=df.index)
            result[has_nan_mask] = np.nan
            
            return result
        
        self.calculate_data[label] = make_label_vec(self.calculate_data)
        
        # 合并数据
        merge_cols = [self.date_column, self.id_column, label] + days_label_list
        self.index_data = pd.merge(self.index_data, self.calculate_data[merge_cols], 
                                   on=[self.date_column, self.id_column], how='left')
        
        # 其余代码保持不变...
        self.calculate_data.to_csv(DataPathConfig.ABORTION_CALCULATE_DATA_SAVE_PATH, index=False, encoding='utf-8-sig')
        self.index_data.to_csv(DataPathConfig.NON_DROP_NAN_LABEL_DATA_SAVE_PATH, index=False, encoding='utf-8-sig')
        
        # 统计标签分布
        total_records = len(self.index_data)
        risk_records = len(self.index_data[self.index_data[label] > 0])
        risk_percentage = (risk_records / total_records) * 100 if total_records > 0 else 0

        # 分布统计
        label_counts = self.index_data[label].value_counts().sort_index()
        logger.info(f"标签分布：")
        for lbl, count in label_counts.items():
            percentage = (count / total_records) * 100
            if lbl == 0:
                logger.info(f"  无风险 (类别 {lbl}): {count} 条记录 ({percentage:.2f}%)")
            else:
                risk_type = "近期(1-7天)" if lbl == 1 else "中期(8-14天)" if lbl == 2 else "远期(15-21天)"
                logger.info(f"  {risk_type} (类别 {lbl}): {count} 条记录 ({percentage:.2f}%)")

        logger.info(f"标签生成完成：总记录数 {total_records}，所有风险记录数 {risk_records} ({risk_percentage:.2f}%)")
        logger.info(f"去除无标签数据前，数据量为：{len(self.index_data)}")
        self.index_data.dropna(subset=[label] + days_label_list, inplace=True)  # 删除没有标签的记录
        logger.info(f"删除无标签的记录后，数据量为: {len(self.index_data)}")

        # 保存带标签的数据
        if DataPathConfig.ABORTION_LABEL_DATA_SAVE_PATH:
            self.index_data.to_csv(DataPathConfig.ABORTION_LABEL_DATA_SAVE_PATH, index=False, encoding='utf-8')
            print(f"带标签的数据已保存至: {DataPathConfig.ABORTION_LABEL_DATA_SAVE_PATH}")

        X = self.index_data.drop(columns=[label] + days_label_list)
        y = self.index_data[['stats_dt', label] + days_label_list]

        return X, y
    
    def has_risk_period_generate_multi_label_alter(self):
        """
        生成标签：如果未来7天内流产率超过0.0025，则标记为1，否则为0
        优化版本 - 修复索引问题和数据处理
        """
        label = ColumnsConfig.HAS_RISK_LABEL
        periods = [(1,7),(8,14),(15,21)]
        days_label_list = [ColumnsConfig.DAYS_RISK_8_CLASS_PRE.format(period[0], period[1]) for period in periods]
        risk_label_list = [ColumnsConfig.HAS_RISK_4_CLASS_PRE.format(period[0], period[1]) for period in periods]

        if self.calculate_data is None or len(self.calculate_data) == 0:
            logger.info("Label计算数据为空，无法生成标签")
            return None
        if self.index_data is None or len(self.index_data) == 0:
            logger.info("INDEX数据为空，无法生成标签")
            return None
        
        # 确保日期列是datetime类型
        self.calculate_data[self.date_column] = pd.to_datetime(self.calculate_data[self.date_column])
        self.index_data[self.date_column] = pd.to_datetime(self.index_data[self.date_column])
        
        # 预初始化所有标签列
        for period in periods:
            left, right = period[0], period[1]
            pre_label = ColumnsConfig.HAS_RISK_4_CLASS_PRE.format(left, right)
            days_label = ColumnsConfig.DAYS_RISK_8_CLASS_PRE.format(left, right)
            self.calculate_data[pre_label] = np.nan
            self.calculate_data[days_label] = np.nan
        
        # 创建一个字典，存储每个猪场的数据，并保留原始索引
        farm_dict = dict(tuple(self.calculate_data.groupby(self.id_column)))
        total_count = len(self.calculate_data)
        processed_count = 0
        
        print("开始处理标签数据...")
        farm_count = 0
        # 处理每个猪场的数据
        for farm_id, farm_data in tqdm(farm_dict.items()):
            farm_count += 1
            # 创建猪场ID和日期到原始索引的映射
            date_to_idx = {}
            for idx, row in farm_data.iterrows():
                date_to_idx[row[self.date_column]] = idx
            
            # 按日期排序，但保持原始索引的引用
            farm_data_sorted = farm_data.sort_values(by=self.date_column)
            farm_dates = farm_data_sorted[self.date_column].values
            farm_rates = farm_data_sorted['abortion_rate'].values
            
            # 处理每个日期记录
            for i, (idx, row) in enumerate(farm_data_sorted.iterrows()):
                current_date = row[self.date_column]
                
                for period in periods:
                    left, right = period[0], period[1]
                    pre_label = ColumnsConfig.HAS_RISK_4_CLASS_PRE.format(left, right)
                    days_label = ColumnsConfig.DAYS_RISK_8_CLASS_PRE.format(left, right)
                    
                    # 计算未来日期范围
                    future_start = current_date + pd.Timedelta(days=left)
                    future_end = current_date + pd.Timedelta(days=right)
                    
                    # 首先检查结束日期是否存在于数据中（确保窗口完整）
                    end_date_exists = np.any(farm_dates == future_end)
                    
                    # 使用向量化操作找出符合日期范围的记录
                    future_mask = (farm_dates >= future_start) & (farm_dates <= future_end)
                    future_rates = farm_rates[future_mask]
                    
                    # 只有当结束日期存在且有足够的数据时才计算标签
                    if end_date_exists and len(future_rates) > 0:
                        # 计算风险标签
                        has_risk = np.any(future_rates >= 0.0025)
                        self.calculate_data.loc[idx, pre_label] = 1 if has_risk else 0
                        
                        # 计算风险天数
                        risk_days = np.sum(future_rates >= 0.0025)
                        self.calculate_data.loc[idx, days_label] = risk_days
        
        # 合并数据
        merge_cols = [self.date_column, self.id_column] + risk_label_list + days_label_list
        self.index_data = pd.merge(self.index_data, self.calculate_data[merge_cols], 
                                   on=[self.date_column, self.id_column], how='left')
        
        # 其余代码保持不变...
        self.calculate_data.to_csv(DataPathConfig.ABORTION_CALCULATE_DATA_SAVE_PATH, index=False, encoding='utf-8-sig')
        self.index_data.to_csv(DataPathConfig.NON_DROP_NAN_LABEL_DATA_SAVE_PATH, index=False, encoding='utf-8-sig')
        
        self.index_data.dropna(subset=risk_label_list + days_label_list, inplace=True)  # 删除没有标签的记录

        # 保存带标签的数据
        if DataPathConfig.ABORTION_LABEL_DATA_SAVE_PATH:
            self.index_data.to_csv(DataPathConfig.ABORTION_LABEL_DATA_SAVE_PATH, index=False, encoding='utf-8')
            print(f"带标签的数据已保存至: {DataPathConfig.ABORTION_LABEL_DATA_SAVE_PATH}")

        X = self.index_data.drop(columns=risk_label_list + days_label_list)
        y = self.index_data[['stats_dt'] + risk_label_list + days_label_list]

        return X, y
    
    def has_risk_4_class_generate_label(self):
        """
        生成标签：如果未来7天内流产率超过0.0025，则标记为1，否则为0
        
        Returns:
            DataFrame: 包含标签列的特征数据
        """
        label = ColumnsConfig.HAS_RISK_LABEL
        periods = [(1,7),(8,14),(15,21)]

        if self.calculate_data is None or len(self.calculate_data) == 0:
            logger.info("Label计算数据为空，无法生成标签")
            return None
        if self.index_data is None or len(self.index_data) == 0:
            logger.info("INDEX数据为空，无法生成标签")
            return None
        
        # 确保日期列是datetime类型
        self.calculate_data[self.date_column] = pd.to_datetime(self.calculate_data[self.date_column])
        self.index_data[self.date_column] = pd.to_datetime(self.index_data[self.date_column])
        
        # 为数据添加标签列，默认为0（无风险）
        # self.feature_data['label'] = 0
        
        # 将数据按猪场ID分组
        grouped_data = self.calculate_data.groupby(self.id_column)
        
        # 对每个猪场处理
        for farm_id, farm_data in tqdm(grouped_data):
            # 确保数据按日期排序
            farm_data = farm_data.sort_values(by=self.date_column)
            
            # 对每个日期记录检查未来7天的风险
            for index, row in farm_data.iterrows():
                current_date = row[self.date_column]
                
                for period in periods:
                    left, right = period[0], period[1]
                    pre_label = ColumnsConfig.HAS_RISK_4_CLASS_PRE.format(left, right)
                    # 计算未来7天的日期范围
                    future_start = current_date + pd.Timedelta(days=left)  # 从下一天开始
                    future_end = current_date + pd.Timedelta(days=right)    # 到7天后结束
                    
                    # 筛选未来7天的数据
                    future_data = farm_data[
                        (farm_data[self.date_column] >= future_start) & 
                        (farm_data[self.date_column] <= future_end)
                    ]
                    
                    # 检查未来7天内是否有流产率超过阈值
                    if not future_data.empty:
                        if (future_data['abortion_rate'] >= 0.0025).any():
                            self.calculate_data.loc[index, pre_label] = 1
                        else:
                            self.calculate_data.loc[index, pre_label] = 0

        def make_label(row):
            make_labels = [ColumnsConfig.HAS_RISK_4_CLASS_PRE.format(period[0], period[1]) for period in periods]
            if row[make_labels[0]] == 1:
                return 1
            elif row[make_labels[1]] == 1:
                return 2
            elif row[make_labels[2]] == 1:
                return 3
            else:
                return 0

        self.calculate_data[label] = self.calculate_data.apply(make_label, axis=1)
        self.index_data = pd.merge(self.index_data, self.calculate_data[[self.date_column, self.id_column, label]], on=[self.date_column, self.id_column], how='left')
        self.calculate_data.to_csv(DataPathConfig.ABORTION_CALCULATE_DATA_SAVE_PATH, index=False, encoding='utf-8-sig')
        self.index_data.to_csv(DataPathConfig.NON_DROP_NAN_LABEL_DATA_SAVE_PATH, index=False, encoding='utf-8-sig')
        
        # 统计标签分布
        total_records = len(self.index_data)
        # 计算非零标签的数量（即风险记录总数 - 类别1+2+3）
        risk_records = len(self.index_data[self.index_data[label] > 0])
        risk_percentage = (risk_records / total_records) * 100 if total_records > 0 else 0

        # 分别统计各类别的记录数和占比
        label_counts = self.index_data[label].value_counts().sort_index()
        logger.info(f"标签分布：")
        for lbl, count in label_counts.items():
            percentage = (count / total_records) * 100
            if lbl == 0:
                logger.info(f"  无风险 (类别 {lbl}): {count} 条记录 ({percentage:.2f}%)")
            else:
                risk_type = "近期(1-7天)" if lbl == 1 else "中期(8-14天)" if lbl == 2 else "远期(15-21天)"
                logger.info(f"  {risk_type} (类别 {lbl}): {count} 条记录 ({percentage:.2f}%)")

        logger.info(f"标签生成完成：总记录数 {total_records}，所有风险记录数 {risk_records} ({risk_percentage:.2f}%)")
        logger.info(f"去除无标签数据前，数据量为：{len(self.index_data)}")
        self.index_data.dropna(subset=[label], inplace=True)  # 删除没有标签的记录
        logger.info(f"删除无标签的记录后，数据量为: {len(self.index_data)}")

        # 保存带标签的数据
        if DataPathConfig.ABORTION_LABEL_DATA_SAVE_PATH:
            self.index_data.to_csv(DataPathConfig.ABORTION_LABEL_DATA_SAVE_PATH, index=False, encoding='utf-8')
            print(f"带标签的数据已保存至: {DataPathConfig.ABORTION_LABEL_DATA_SAVE_PATH}")

        X = self.index_data.drop(columns=[label])
        y = self.index_data[['stats_dt', label]]

        return X, y
    
    def has_risk_3_point_generate_label(self):
        """
        生成标签：如果未来7天内流产率超过0.0025，则标记为1，否则为0
        
        Returns:
            DataFrame: 包含标签列的特征数据
        """
        label = ColumnsConfig.HAS_RISK_3_POINT_LABEL
        periods = [7,14,21]

        if self.calculate_data is None or len(self.calculate_data) == 0:
            logger.info("Label计算数据为空，无法生成标签")
            return None
        if self.index_data is None or len(self.index_data) == 0:
            logger.info("INDEX数据为空，无法生成标签")
            return None
        
        # 确保日期列是datetime类型
        self.calculate_data[self.date_column] = pd.to_datetime(self.calculate_data[self.date_column])
        self.index_data[self.date_column] = pd.to_datetime(self.index_data[self.date_column])
        
        # 为数据添加标签列，默认为0（无风险）
        # self.feature_data['label'] = 0
        
        # 将数据按猪场ID分组
        grouped_data = self.calculate_data.groupby(self.id_column)

        for farm_id, farm_data in tqdm(grouped_data):
            # 确保数据按日期排序
            farm_data = farm_data.sort_values(by=self.date_column)
            
            # 对每个日期记录检查未来7天的风险
            for index, row in farm_data.iterrows():
                current_date = row[self.date_column]
                
                for period in periods:
                    pre_label = "future_{}_label".format(period)
                    # 计算未来7天的日期范围
                    future_point = current_date + pd.Timedelta(days=period)    # 到7天后结束
                    
                    # 筛选未来7天的数据
                    future_data = farm_data[farm_data[self.date_column] == future_point]
                    
                    # 检查未来7天内是否有流产率超过阈值
                    if not future_data.empty:
                        if (future_data['abortion_rate'].iloc[0] >= 0.0025):
                            self.calculate_data.loc[index, pre_label] = 1
                        else:
                            self.calculate_data.loc[index, pre_label] = 0

        self.index_data = pd.merge(self.index_data, self.calculate_data[[self.date_column, self.id_column] + ColumnsConfig.HAS_RISK_3_POINT_LABEL], on=[self.date_column, self.id_column], how='left')
        self.calculate_data.to_csv(DataPathConfig.ABORTION_CALCULATE_DATA_SAVE_PATH, index=False, encoding='utf-8-sig')
        self.index_data.to_csv(DataPathConfig.NON_DROP_NAN_LABEL_DATA_SAVE_PATH, index=False, encoding='utf-8-sig')
        
        # 统计标签分布
        total_records = len(self.index_data)
        
        # 计算任一标签为1的记录数（即任何时间点有风险的记录）
        any_risk_records = len(self.index_data[(self.index_data[label[0]] == 1) | 
                                              (self.index_data[label[1]] == 1) | 
                                              (self.index_data[label[2]] == 1)])
        any_risk_percentage = (any_risk_records / total_records) * 100 if total_records > 0 else 0
        
        logger.info(f"标签分布：")
        # 分别统计每个标签的分布
        for i, lbl in enumerate(label):
            day = periods[i]  # 7, 14, 或 21
            risk_records = self.index_data[lbl].sum()
            risk_percentage = (risk_records / total_records) * 100 if total_records > 0 else 0
            
            logger.info(f"  未来第{day}天: 有风险记录 {risk_records} 条 ({risk_percentage:.2f}%), "
                        f"无风险记录 {total_records - risk_records} 条 ({100 - risk_percentage:.2f}%)")
        
        logger.info(f"标签生成完成：总记录数 {total_records}，任一时间点有风险的记录数 {any_risk_records} ({any_risk_percentage:.2f}%)")

        logger.info(f"去除无标签数据前，数据量为：{len(self.index_data)}")
        self.index_data.dropna(subset=label, inplace=True, how='any')  # 删除没有标签的记录
        logger.info(f"删除无标签的记录后，数据量为: {len(self.index_data)}")

        # 保存带标签的数据
        if DataPathConfig.ABORTION_LABEL_DATA_SAVE_PATH:
            self.index_data.to_csv(DataPathConfig.ABORTION_LABEL_DATA_SAVE_PATH, index=False, encoding='utf-8')
            print(f"带标签的数据已保存至: {DataPathConfig.ABORTION_LABEL_DATA_SAVE_PATH}")

        X = self.index_data.drop(columns=label)
        y = self.index_data[['stats_dt'] + label]

        return X, y
    
    # def gen_abortion_predict_label(self):

    #     return X, y
