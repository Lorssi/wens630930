import pandas as pd
from configs.feature_config import DataPathConfig,ColumnsConfig
from configs.logger_config import logger_config
from configs.base_config import FeatureData
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
            df = pd.read_csv(FeatureData.PRODUCTION_FEATURE_DATA.value, encoding='utf-8')
            logger.info(f"成功加载数据: {FeatureData.PRODUCTION_FEATURE_DATA.value}, 数据行数: {len(df)}")

            df[self.date_column] = pd.to_datetime(df[self.date_column])  # 转换为 datetime 格式
            df[self.date_column] = df[self.date_column] - pd.DateOffset(days=1) # 保存在本地的生产数据时间是+1过的，这里要-1变回原始时间
            df = df.dropna(subset=['abortion_rate'])  # 删除缺失值行
            df = df[[self.date_column, self.id_column, 'abortion_rate']]  # 只保留需要的列

            # 处理数据
            if df is not None:
                # 确保在计算前，数据是按猪场和日期排序的
                df.sort_values(by=[self.date_column, self.id_column], inplace=True)

            return df
        except FileNotFoundError:
            logger.error(f"错误: 数据文件未找到于 {DataPathConfig.ML_AOBRTION_RATE_DATA_SAVE_PATH}")
            return None
        

    def has_risk_period_generate_multi_label_alter_nodays(self):
        """
        生成标签：如果未来7天内流产率超过0.0025，则标记为1，否则为0
        修改版本 - 移除完整窗口限制:
        1. 对于完整窗口: 正常计算标签(1或0)
        2. 对于不完整窗口: 如果观察到流产率>=0.0025则标记为1，否则为NaN
        """
        periods = [(1,7),(8,14),(15,21)]
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
            self.calculate_data[pre_label] = np.nan
        
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
            
            # 获取最大日期，用于判断窗口是否完整
            max_date = farm_dates.max()
            
            # 处理每个日期记录
            for i, (idx, row) in enumerate(farm_data_sorted.iterrows()):
                current_date = row[self.date_column]
                
                for period in periods:
                    left, right = period[0], period[1]
                    pre_label = ColumnsConfig.HAS_RISK_4_CLASS_PRE.format(left, right)
                    
                    # 计算未来日期范围
                    future_start = current_date + pd.Timedelta(days=left)
                    future_end = current_date + pd.Timedelta(days=right)
                    
                    # 检查窗口是否完整(窗口终点是否在数据范围内)
                    window_complete = future_end <= max_date
                    
                    # 使用向量化操作找出符合日期范围的记录
                    future_mask = (farm_dates >= future_start) & (farm_dates <= future_end)
                    future_rates = farm_rates[future_mask]
                    
                    # 检查窗口内是否有数据
                    if len(future_rates) > 0:
                        # 检查是否有流产率大于阈值
                        has_risk = np.any(future_rates >= 0.0025)
                        
                        if has_risk:
                            # 如果检测到风险，则标记为1
                            self.calculate_data.loc[idx, pre_label] = 1
                            
                            # days_label还按固定完整窗口来计算
                            if window_complete:
                                risk_days = np.sum(future_rates >= 0.0025)
                        elif window_complete:
                            # 如果窗口完整且没有风险，则标记为0
                            self.calculate_data.loc[idx, pre_label] = 0
                        # 如果窗口不完整且没有风险，保持为NaN
    
        # 合并数据
        merge_cols = [self.date_column, self.id_column] + risk_label_list
        self.index_data = pd.merge(self.index_data, self.calculate_data[merge_cols], 
                                   on=[self.date_column, self.id_column], how='left')
        
        # 保存数据
        self.calculate_data.to_csv(DataPathConfig.ABORTION_CALCULATE_DATA_SAVE_PATH, index=False, encoding='utf-8-sig')
        self.index_data.to_csv(DataPathConfig.NON_DROP_NAN_LABEL_DATA_SAVE_PATH, index=False, encoding='utf-8-sig')
        
        # 统计标签信息
        for period in periods:
            left, right = period[0], period[1]
            pre_label = ColumnsConfig.HAS_RISK_4_CLASS_PRE.format(left, right)
            total = len(self.index_data)
            nan_count = self.index_data[pre_label].isna().sum()
            valid_count = total - nan_count
            risk_count = (self.index_data[pre_label] == 1).sum()
            
            if valid_count > 0:
                risk_pct = (risk_count / valid_count) * 100
                logger.info(f"{left}-{right}天窗口: 共{total}条, 有效{valid_count}条, 风险{risk_count}条 ({risk_pct:.2f}%)")
        
        # 删除标签有为NaN的记录
        self.index_data.dropna(subset=risk_label_list, how='any', inplace=True)

        # 保存带标签的数据
        if DataPathConfig.ABORTION_LABEL_DATA_SAVE_PATH:
            self.index_data.to_csv(DataPathConfig.ABORTION_LABEL_DATA_SAVE_PATH, index=False, encoding='utf-8')
            print(f"带标签的数据已保存至: {DataPathConfig.ABORTION_LABEL_DATA_SAVE_PATH}")

        X = self.index_data.drop(columns=risk_label_list)
        y = self.index_data[risk_label_list]

        return X, y
    
    def has_risk_period_generate_multi_label_alter_risk_days(self):
        """
        生成标签：如果未来7天内流产率超过0.0025，则标记为1，否则为0
        修改版本 - 移除完整窗口限制:
        1. 对于完整窗口: 正常计算标签(1或0)
        2. 对于不完整窗口: 如果观察到流产率>=0.0025则标记为1，否则为NaN
        """
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
            risk_label = ColumnsConfig.HAS_RISK_4_CLASS_PRE.format(left, right)
            days_label = ColumnsConfig.DAYS_RISK_8_CLASS_PRE.format(left, right)
            self.calculate_data[risk_label] = np.nan
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
            
            # 获取最大日期，用于判断窗口是否完整
            max_date = farm_dates.max()
            
            # 处理每个日期记录
            for i, (idx, row) in enumerate(farm_data_sorted.iterrows()):
                current_date = row[self.date_column]
                
                for period in periods:
                    left, right = period[0], period[1]
                    risk_label = ColumnsConfig.HAS_RISK_4_CLASS_PRE.format(left, right)
                    days_label = ColumnsConfig.DAYS_RISK_8_CLASS_PRE.format(left, right)
                    
                    # 计算未来日期范围
                    future_start = current_date + pd.Timedelta(days=left)
                    future_end = current_date + pd.Timedelta(days=right)
                    
                    # 检查窗口是否完整(窗口终点是否在数据范围内)
                    window_complete = future_end <= max_date
                    
                    # 使用向量化操作找出符合日期范围的记录
                    future_mask = (farm_dates >= future_start) & (farm_dates <= future_end)
                    future_rates = farm_rates[future_mask]
                    
                    # 检查窗口内是否有数据
                    if len(future_rates) > 0:
                        # 检查是否有流产率大于阈值
                        has_risk = np.any(future_rates >= 0.0025)
                        
                        if has_risk:
                            # 如果检测到风险，则标记为1
                            self.calculate_data.loc[idx, risk_label] = 1
                            
                            # days_label还按固定完整窗口来计算
                            if window_complete:
                                risk_days = np.sum(future_rates >= 0.0025)
                                self.calculate_data.loc[idx, days_label] = risk_days
                        elif window_complete:
                            # 如果窗口完整且没有风险，则标记为0
                            self.calculate_data.loc[idx, risk_label] = 0
                            self.calculate_data.loc[idx, days_label] = 0
                        # 如果窗口不完整且没有风险，保持为NaN
    
        # 合并数据
        merge_cols = [self.date_column, self.id_column] + risk_label_list + days_label_list
        self.index_data = pd.merge(self.index_data, self.calculate_data[merge_cols], 
                                   on=[self.date_column, self.id_column], how='left')
        
        # 保存数据
        self.calculate_data.to_csv(DataPathConfig.ABORTION_CALCULATE_DATA_SAVE_PATH, index=False, encoding='utf-8-sig')
        self.index_data.to_csv(DataPathConfig.NON_DROP_NAN_LABEL_DATA_SAVE_PATH, index=False, encoding='utf-8-sig')
        
        # 删除标签有为NaN的记录
        self.index_data.dropna(subset=risk_label_list+days_label_list, how='any', inplace=True)

        # 保存带标签的数据
        if DataPathConfig.ABORTION_LABEL_DATA_SAVE_PATH:
            self.index_data.to_csv(DataPathConfig.ABORTION_LABEL_DATA_SAVE_PATH, index=False, encoding='utf-8')
            print(f"带标签的数据已保存至: {DataPathConfig.ABORTION_LABEL_DATA_SAVE_PATH}")

        X = self.index_data.drop(columns=days_label_list+risk_label_list)
        y = self.index_data[days_label_list+risk_label_list]

        return X, y
    
    def has_risk_period_generate_multi_label_days_alter(self):
        """
        生成标签：如果未来7天内流产率超过0.0025，则标记为1，否则为0
        修改版本 - 移除完整窗口限制:
        1. 对于完整窗口: 正常计算标签(1或0)
        2. 对于不完整窗口: 如果观察到流产率>=0.0025则标记为1，否则为NaN
        """
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
            
            # 获取最大日期，用于判断窗口是否完整
            max_date = farm_dates.max()
            
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
                    
                    # 检查窗口是否完整(窗口终点是否在数据范围内)
                    window_complete = future_end <= max_date
                    
                    # 使用向量化操作找出符合日期范围的记录
                    future_mask = (farm_dates >= future_start) & (farm_dates <= future_end)
                    future_rates = farm_rates[future_mask]
                    
                    # 检查窗口内是否有数据
                    if len(future_rates) > 0:
                        # 检查是否有流产率大于阈值
                        has_risk = np.any(future_rates >= 0.0025)
                        
                        if has_risk:
                            # 如果检测到风险，则标记为1
                            self.calculate_data.loc[idx, pre_label] = 1
                            
                            # days_label还按固定完整窗口来计算
                            if window_complete:
                                risk_days = np.sum(future_rates >= 0.0025)
                                self.calculate_data.loc[idx, days_label] = risk_days
                        elif window_complete:
                            # 如果窗口完整且没有风险，则标记为0
                            self.calculate_data.loc[idx, pre_label] = 0
                            self.calculate_data.loc[idx, days_label] = 0
                        # 如果窗口不完整且没有风险，保持为NaN
    
        # 合并数据
        merge_cols = [self.date_column, self.id_column] + days_label_list
        self.index_data = pd.merge(self.index_data, self.calculate_data[merge_cols], 
                                   on=[self.date_column, self.id_column], how='left')
        
        # 保存数据
        self.calculate_data.to_csv(DataPathConfig.ABORTION_CALCULATE_DATA_SAVE_PATH, index=False, encoding='utf-8-sig')
        self.index_data.to_csv(DataPathConfig.NON_DROP_NAN_LABEL_DATA_SAVE_PATH, index=False, encoding='utf-8-sig')
        
        # 删除标签有为NaN的记录
        self.index_data.dropna(subset=days_label_list, how='any', inplace=True)

        # 保存带标签的数据
        if DataPathConfig.ABORTION_LABEL_DATA_SAVE_PATH:
            self.index_data.to_csv(DataPathConfig.ABORTION_LABEL_DATA_SAVE_PATH, index=False, encoding='utf-8')
            print(f"带标签的数据已保存至: {DataPathConfig.ABORTION_LABEL_DATA_SAVE_PATH}")

        X = self.index_data.drop(columns=days_label_list)
        y = self.index_data[days_label_list]

        return X, y
    

    
    def abortion_predict_gen_sequence_label(self):
        """
        为序列预测生成标签：将特征数据扩展为每条记录21行（预测未来1-21天），
        并从calculate_data中获取对应日期的流产率作为标签
        
        优化版本 - 使用列表存储避免类型转换问题，同时提高处理速度
        
        Returns:
            tuple: (X, y) 其中X是特征数据，y是标签数据
        """
        if self.calculate_data is None or len(self.calculate_data) == 0:
            logger.info("Label计算数据为空，无法生成标签")
            return None, None
        if self.index_data is None or len(self.index_data) == 0:
            logger.info("INDEX数据为空，无法生成标签")
            return None, None
        
        # 确保日期列是datetime类型
        self.calculate_data[self.date_column] = pd.to_datetime(self.calculate_data[self.date_column])
        self.index_data[self.date_column] = pd.to_datetime(self.index_data[self.date_column])
        
        logger.info("开始为序列预测生成标签数据...")
        
        # 1. 创建一个高效的查询结构 - 使用字典按猪场ID分组
        logger.info("构建查询索引...")
        farm_abortion_rates = {}
        for farm_id, group in tqdm(self.calculate_data.groupby(self.id_column), desc="构建索引"):
            date_rates = {}
            for _, row in group.iterrows():
                date_rates[row[self.date_column].date()] = row['abortion_rate']
            farm_abortion_rates[farm_id] = date_rates
        
        # 2. 使用列表存储结果，避免NumPy数组的类型转换问题
        new_rows = []
        new_rows_append = new_rows.append  # 缓存方法查找
        
        # 3. 批量处理数据生成
        total_records = 0
        valid_records = 0
        
        # 4. 按猪场分组处理，提高局部性能
        for farm_id, farm_group in tqdm(self.index_data.groupby(self.id_column), desc="生成序列预测数据"):
            if farm_id not in farm_abortion_rates:
                continue
                
            farm_dates = farm_abortion_rates[farm_id]
            
            for _, row in farm_group.iterrows():
                base_date = row[self.date_column].date()
                row_dict = row.to_dict()  # 转为字典，提高后续处理速度
                
                # 批量检查所有目标日期
                for day_offset in range(1, 22):
                    target_date = base_date + pd.Timedelta(days=day_offset)
                    total_records += 1
                    
                    if target_date in farm_dates:
                        # 创建新行 (使用字典复制更快)
                        new_row = row_dict.copy()
                        new_row['predict_date'] = day_offset
                        new_row['target_abortion_rate'] = farm_dates[target_date]
                        
                        new_rows_append(new_row)
                        valid_records += 1
        
        # 如果没有有效数据
        if not new_rows:
            logger.info("没有有效的预测数据")
            return None, None
        
        # 创建DataFrame
        expanded_data = pd.DataFrame(new_rows)
        
        logger.info(f"序列预测数据生成完成：总共尝试生成 {total_records} 条记录，生成有效记录 {valid_records} 条")
        
        # 分离特征和标签
        y = expanded_data[['stats_dt', 'predict_date', 'target_abortion_rate']]
        X = expanded_data.drop(columns=['target_abortion_rate'])
        
        # 保存带标签的数据
        if DataPathConfig.ABORTION_LABEL_DATA_SAVE_PATH:
            self.index_data.to_csv(DataPathConfig.ABORTION_LABEL_DATA_SAVE_PATH, index=False, encoding='utf-8')
            print(f"带标签的数据已保存至: {DataPathConfig.ABORTION_LABEL_DATA_SAVE_PATH}")
        
        return X, y
    
    def has_risk_period_generate_multi_label_alter_nodays_expand14(self):
        """
        生成标签：使用14天窗口判断是否存在风险
        对于当前样本时间T：
        - 1-7天标签：检查[T-6, T+7]窗口内是否有流产率>0.0025
        - 8-14天标签：检查[T+1, T+14]窗口内是否有流产率>0.0025
        - 15-21天标签：检查[T+8, T+21]窗口内是否有流产率>0.0025
        """
        periods = [(1,7),(8,14),(15,21)]
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
            self.calculate_data[pre_label] = np.nan
        
        # 创建一个字典，存储每个猪场的数据，并保留原始索引
        farm_dict = dict(tuple(self.calculate_data.groupby(self.id_column)))
        
        print("开始处理标签数据...")
        # 处理每个猪场的数据
        for farm_id, farm_data in tqdm(farm_dict.items()):
            # 按日期排序，但保持原始索引的引用
            farm_data_sorted = farm_data.sort_values(by=self.date_column)
            farm_dates = farm_data_sorted[self.date_column].values
            farm_rates = farm_data_sorted['abortion_rate'].values
            
            # 获取最大日期，用于判断窗口是否完整
            max_date = farm_dates.max()
            
            # 处理每个日期记录
            for idx, row in farm_data_sorted.iterrows():
                current_date = row[self.date_column]
                
                for period in periods:
                    left, right = period[0], period[1]
                    pre_label = ColumnsConfig.HAS_RISK_4_CLASS_PRE.format(left, right)
                    
                    # 根据不同时段计算14天窗口范围
                    if left == 1 and right == 7:  # 1-7天时段
                        future_start = current_date - pd.Timedelta(days=6)  # T-6
                        future_end = current_date + pd.Timedelta(days=7)    # T+7
                    elif left == 8 and right == 14:  # 8-14天时段
                        future_start = current_date + pd.Timedelta(days=1)  # T+1
                        future_end = current_date + pd.Timedelta(days=14)   # T+14
                    elif left == 15 and right == 21:  # 15-21天时段
                        future_start = current_date + pd.Timedelta(days=8)  # T+8
                        future_end = current_date + pd.Timedelta(days=21)   # T+21
                    
                    # 检查窗口是否完整(窗口终点是否在数据范围内)
                    window_complete = future_end <= max_date
                    
                    # 使用向量化操作找出符合日期范围的记录
                    future_mask = (farm_dates >= future_start) & (farm_dates <= future_end)
                    future_rates = farm_rates[future_mask]
                    
                    # 检查窗口内是否有数据
                    if len(future_rates) > 0:
                        # 检查是否有流产率大于阈值
                        has_risk = np.any(future_rates >= 0.0025)
                        
                        if has_risk:
                            # 如果检测到风险，则标记为1
                            self.calculate_data.loc[idx, pre_label] = 1
                        elif window_complete:
                            # 如果窗口完整且没有风险，则标记为0
                            self.calculate_data.loc[idx, pre_label] = 0
                        # 如果窗口不完整且没有风险，保持为NaN

        # 合并数据
        merge_cols = [self.date_column, self.id_column] + risk_label_list
        self.index_data = pd.merge(self.index_data, self.calculate_data[merge_cols], 
                                   on=[self.date_column, self.id_column], how='left')
        
        # 保存数据
        self.calculate_data.to_csv(DataPathConfig.ABORTION_CALCULATE_DATA_SAVE_PATH, index=False, encoding='utf-8-sig')
        self.index_data.to_csv(DataPathConfig.NON_DROP_NAN_LABEL_DATA_SAVE_PATH, index=False, encoding='utf-8-sig')
        
        # 统计标签信息
        for period in periods:
            left, right = period[0], period[1]
            pre_label = ColumnsConfig.HAS_RISK_4_CLASS_PRE.format(left, right)
            total = len(self.index_data)
            nan_count = self.index_data[pre_label].isna().sum()
            valid_count = total - nan_count
            risk_count = (self.index_data[pre_label] == 1).sum()
            
            if valid_count > 0:
                risk_pct = (risk_count / valid_count) * 100
                logger.info(f"{left}-{right}天窗口: 共{total}条, 有效{valid_count}条, 风险{risk_count}条 ({risk_pct:.2f}%)")
        
        # 删除标签有为NaN的记录
        self.index_data.dropna(subset=risk_label_list, how='any', inplace=True)

        # 保存带标签的数据
        if DataPathConfig.ABORTION_LABEL_DATA_SAVE_PATH:
            self.index_data.to_csv(DataPathConfig.ABORTION_LABEL_DATA_SAVE_PATH, index=False, encoding='utf-8')
            print(f"带标签的数据已保存至: {DataPathConfig.ABORTION_LABEL_DATA_SAVE_PATH}")

        X = self.index_data.drop(columns=risk_label_list)
        y = self.index_data[risk_label_list]

        return X, y
    
    def has_risk_period_generate_multi_label_alter_nodays_continuous(self):
        """
        生成标签：如果未来对应时段内流产率超过0.0025，则标记为1，否则为0
        修改版本 - 风险递增处理:
        1. 对于风险区间(值为1)之前的最多7个无风险样本(值为0)
        2. 将这些样本的标签修改为从0到1递增的浮点数(0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 0.9)
        3. 表示随着时间接近风险期，风险逐渐上升
        """
        periods = [(1,7),(8,14),(15,21)]
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
            self.calculate_data[pre_label] = np.nan
    
        # 创建一个字典，存储每个猪场的数据，并保留原始索引
        farm_dict = dict(tuple(self.calculate_data.groupby(self.id_column)))
        total_count = len(self.calculate_data)
        
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
        
            # 获取最大日期，用于判断窗口是否完整
            max_date = farm_dates.max()
        
            # 处理每个日期记录
            for i, (idx, row) in enumerate(farm_data_sorted.iterrows()):
                current_date = row[self.date_column]
                
                for period in periods:
                    left, right = period[0], period[1]
                    pre_label = ColumnsConfig.HAS_RISK_4_CLASS_PRE.format(left, right)
                    
                    # 计算未来日期范围
                    future_start = current_date + pd.Timedelta(days=left)
                    future_end = current_date + pd.Timedelta(days=right)
                    
                    # 检查窗口是否完整(窗口终点是否在数据范围内)
                    window_complete = future_end <= max_date
                    
                    # 使用向量化操作找出符合日期范围的记录
                    future_mask = (farm_dates >= future_start) & (farm_dates <= future_end)
                    future_rates = farm_rates[future_mask]
                    
                    # 检查窗口内是否有数据
                    if len(future_rates) > 0:
                        # 检查是否有流产率大于阈值
                        has_risk = np.any(future_rates >= 0.0025)
                        
                        if has_risk:
                            # 如果检测到风险，则标记为1
                            self.calculate_data.loc[idx, pre_label] = 1
                        elif window_complete:
                            # 如果窗口完整且没有风险，则标记为0
                            self.calculate_data.loc[idx, pre_label] = 0
                        # 如果窗口不完整且没有风险，保持为NaN

        # 第二阶段：为每个猪场的每个时段添加风险递增标签
        print("开始处理风险递增标签...")
        
        # 定义递增值序列（最多前7个样本）
        risk_increase_values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 0.9]
        
        for farm_id, farm_data in tqdm(farm_dict.items()):
            # 按日期排序处理每个猪场数据
            farm_indices = farm_data.index.tolist()
            farm_dates = farm_data[self.date_column].tolist()
            
            # 确保日期和索引是按照日期排序的
            sorted_indices = [x for _, x in sorted(zip(farm_dates, farm_indices))]
            
            for period in periods:
                left, right = period[0], period[1]
                pre_label = ColumnsConfig.HAS_RISK_4_CLASS_PRE.format(left, right)
                
                for period_idx in range(len(sorted_indices)):
                    current_idx = sorted_indices[period_idx]
                    current_value = self.calculate_data.loc[current_idx, pre_label]
                    
                    # 如果当前值是1，说明这是风险区间的开始
                    if current_value == 1:
                        # 向前最多查找7个样本
                        look_back = min(7, period_idx)
                        
                        # 检查前面最多7个样本
                        for back_offset in range(1, look_back + 1):
                            prev_idx = sorted_indices[period_idx - back_offset]
                            prev_value = self.calculate_data.loc[prev_idx, pre_label]
                            
                            # 如果前面的样本是0，则修改为递增的风险值
                            if prev_value == 0:
                                # 计算递增值索引，从后往前，最近的为0.9，往前依次递减
                                increase_idx = back_offset - 1
                                risk_value = risk_increase_values[-(back_offset)]
                                self.calculate_data.loc[prev_idx, pre_label] = risk_value
                            else:
                                # 如果不是0，说明可能已经被前面的风险期处理过
                                # 或者是NaN，就不再继续向前处理
                                break
        
        # 合并数据
        merge_cols = [self.date_column, self.id_column] + risk_label_list
        self.index_data = pd.merge(self.index_data, self.calculate_data[merge_cols], 
                                   on=[self.date_column, self.id_column], how='left')
        
        # 保存数据
        self.calculate_data.to_csv(DataPathConfig.ABORTION_CALCULATE_DATA_SAVE_PATH, index=False, encoding='utf-8-sig')
        self.index_data.to_csv(DataPathConfig.NON_DROP_NAN_LABEL_DATA_SAVE_PATH, index=False, encoding='utf-8-sig')
        
        # 统计标签信息
        for period in periods:
            left, right = period[0], period[1]
            pre_label = ColumnsConfig.HAS_RISK_4_CLASS_PRE.format(left, right)
            total = len(self.index_data)
            nan_count = self.index_data[pre_label].isna().sum()
            valid_count = total - nan_count
            risk_count = (self.index_data[pre_label] == 1).sum()
            risk_increasing_count = ((self.index_data[pre_label] > 0) & (self.index_data[pre_label] < 1)).sum()
            
            if valid_count > 0:
                risk_pct = (risk_count / valid_count) * 100
                risk_inc_pct = (risk_increasing_count / valid_count) * 100
                logger.info(f"{left}-{right}天窗口: 共{total}条, 有效{valid_count}条, 风险{risk_count}条 ({risk_pct:.2f}%), "
                            f"风险递增{risk_increasing_count}条 ({risk_inc_pct:.2f}%)")
        
        # 删除标签有为NaN的记录
        self.index_data.dropna(subset=risk_label_list, how='any', inplace=True)

        # 保存带标签的数据
        if DataPathConfig.ABORTION_LABEL_DATA_SAVE_PATH:
            self.index_data.to_csv(DataPathConfig.ABORTION_LABEL_DATA_SAVE_PATH, index=False, encoding='utf-8')
            print(f"带标签的数据已保存至: {DataPathConfig.ABORTION_LABEL_DATA_SAVE_PATH}")

        X = self.index_data.drop(columns=risk_label_list)
        y = self.index_data[risk_label_list]

        return X, y
    
    def has_risk_period_generate_multi_label_alter_nodays_fake_abnormal(self, fake_abnormal_num = 4):
        """
        生成标签后对数据进行处理：将每个连续1序列之前的4个0修改为1
        
        Returns:
            tuple: (X, y) 特征和标签数据
        """
        # 生成原始标签数据
        X, y = self.has_risk_period_generate_multi_label_alter_nodays()
        data = pd.concat([X, y], axis=1).reset_index(drop=True)
        
        # 获取三个时段的标签列名
        periods = [(1,7), (8,14), (15,21)]
        risk_label_list = [ColumnsConfig.HAS_RISK_4_CLASS_PRE.format(period[0], period[1]) for period in periods]
        
        print("开始处理连续1之前的标签...")
        
        # 对每个标签列处理
        for label_col in risk_label_list:
            print(f"处理标签列: {label_col}")
            
            # 按猪场ID分组处理
            for farm_id, farm_group in tqdm(data.groupby(self.id_column), desc=f"处理{label_col}"):
                # 按日期排序
                farm_data = farm_group.sort_values(by=self.date_column).copy()
                
                # 获取标签值和索引
                farm_indices = farm_data.index.tolist()
                farm_labels = farm_data[label_col].tolist()
                
                # 用于标记哪些位置需要修改
                to_modify = []
                
                # 用于识别连续1的开始
                in_ones_sequence = False
                
                # 从头到尾扫描数据
                for i, (idx, val) in enumerate(zip(farm_indices, farm_labels)):
                    # 检测连续1的开始
                    if val == 1 and not in_ones_sequence:
                        in_ones_sequence = True
                        
                        # 向前寻找最多4个0
                        count = 0
                        for j in range(i-1, -1, -1):
                            if count >= fake_abnormal_num:
                                break
                            if farm_labels[j] == 0:
                                to_modify.append(farm_indices[j])
                                count += 1
                    
                    # 检测连续1的结束
                    elif val != 1 and in_ones_sequence:
                        in_ones_sequence = False
                
                # 批量修改
                if to_modify:
                    data.loc[to_modify, label_col] = 1

        # 保存带标签的数据
        if DataPathConfig.ABORTION_LABEL_DATA_SAVE_PATH:
            data.to_csv(DataPathConfig.ABORTION_LABEL_DATA_SAVE_PATH, index=False, encoding='utf-8')
            print(f"带标签的数据已保存至: {DataPathConfig.ABORTION_LABEL_DATA_SAVE_PATH}")
        
        print("处理完成，统计修改后的标签分布...")
        
        # 统计修改后的标签分布
        for period, label_col in zip(periods, risk_label_list):
            left, right = period
            total = len(data)
            risk_count = (data[label_col] == 1).sum()
            risk_pct = (risk_count / total) * 100
            print(f"{left}-{right}天窗口: 共{total}条, 风险{risk_count}条 ({risk_pct:.2f}%)")
        
        # 分离特征和标签
        X = data.drop(columns=risk_label_list)
        y = data[risk_label_list]
        
        return X, y



