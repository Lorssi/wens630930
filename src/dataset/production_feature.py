import os
import sys
import logging
from pathlib import Path

import pandas as pd
import numpy as np
from tqdm import tqdm

if 'd:\\data\\VSCode\\wens630930\\src\\dataset' in sys.path:
    sys.path.remove('d:\\data\\VSCode\\wens630930\\src\\dataset')
    # 获取项目根目录
    project_root = Path(__file__).resolve().parent.parent
    sys.path.append(str(project_root))

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


class ProductionFeature(BaseFeatureDataSet):

    def __init__(self, running_dt_end: str, train_interval: int, file_type: str, **param):
        super().__init__(param)
        logger.info('-----Loading data-----')
        self.production_data = pd.read_csv(RawData.ADS_PIG_ORG_TOTAL_TO_ML_TRAINING_DAY.value, encoding='utf-8')
        self.production_line_data = pd.read_csv(RawData.ADS_PIG_PROLINE_TOTAL_TO_ML_TRAINING_DAY.value, encoding='utf-8')

        self.running_dt_end = running_dt_end
        self.train_interval = train_interval
        self.file_type = file_type

        self.end_date = (datetime.strptime(self.running_dt_end, "%Y-%m-%d") - timedelta(days=1)).strftime("%Y-%m-%d")
        self.start_date = (datetime.strptime(self.end_date, "%Y-%m-%d") - timedelta(days=self.train_interval) - timedelta(days=40)).strftime("%Y-%m-%d")
        logger.info('-----start_date: {}'.format(self.start_date))
        logger.info('-----end_date: {}'.format(self.end_date))

        self.data = pd.DataFrame()
        self.file_name = None

        self._init_entity_and_features()

    def _preprocessing_data(self):
        production_data = self.production_data.copy()

        production_data['stats_dt'] = pd.to_datetime(production_data['stats_dt'])
        production_data = production_data[
            (production_data['stats_dt'] >= self.start_date) &
            (production_data['stats_dt'] <= self.end_date)]

        production_data.sort_values(['stats_dt', 'pigfarm_dk'], inplace=True)
        # 更新
        self.production_data = production_data

        production_line_data = self.production_line_data.copy()
        production_line_data['stats_dt'] = pd.to_datetime(production_line_data['stats_dt'])
        production_line_data = production_line_data[
            (production_line_data['stats_dt'] >= self.start_date) &
            (production_line_data['stats_dt'] <= self.end_date)]
        production_line_data.sort_values(['stats_dt', 'pigfarm_dk'], inplace=True)
        self.production_line_data = production_line_data

    def _init_entity_and_features(self):
        """初始化实体和特征定义"""
        # 设置实体列表
        self.entity = ['stats_dt', 'pigfarm_dk']
        # 定义特征
        features_config = [
            # 组织特征
            ('pigfarm_dk', FeatureType.Categorical, FeatureDtype.String, 'organization'),
            # 流产率特征
            ('abortion_rate', FeatureType.Continuous, FeatureDtype.Float32, 'abortion'),
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

    def _get_abortion_rate(self):
        """
        计算流产率。
        流产率 = sum(近7天流产数量) / (sum(近7天流产数量) + 当天怀孕母猪存栏量)
        """
        abort_qty_column = 'abort_qty'
        preg_stock_qty_column = 'preg_stock_qty'
        id_column = 'pigfarm_dk'
        date_column = 'stats_dt'

        feature_name = 'abortion_rate'

        production_data = self.production_data.copy()

        # 确保流产数量和怀孕母猪存栏量是数值类型，并将NaN填充为0，因为它们参与计算
        production_data[abort_qty_column] = pd.to_numeric(production_data[abort_qty_column], errors='coerce').fillna(0)
        production_data[preg_stock_qty_column] = pd.to_numeric(production_data[preg_stock_qty_column], errors='coerce').fillna(0)

        # 使用 groupby 和 rolling window 计算每个猪场每个日期的近7天流产总数
        # 'closed="left"' 通常用于rolling sum，但这里我们需要包含当天，所以默认'right'就可以
        # min_periods=1 表示即使不足7天，也会计算已有的天数和
        production_data['recent_7day_abort_sum'] = production_data.groupby(id_column)[abort_qty_column]\
                                                            .rolling(window=7, min_periods=7).sum()\
                                                            .reset_index(level=0, drop=True) # reset_index 去掉 groupby 带来的多级索引

        # 定义一个函数来计算流产率，处理分母为0的情况
        def calculate_rate(row):
            sum_recent_abort = row['recent_7day_abort_sum']
            current_preg_stock = row[preg_stock_qty_column]

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
        production_data[feature_name] = production_data.apply(calculate_rate, axis=1)

        self.data = production_data.copy()

    def _get_abortion_mean_feature(self):
        """
        计算流产率特征:
        1. abortion_feature_1_7: 流产率
        2. abortion_mean_recent_7d: 近7天流产率均值
        3. abortion_mean_recent_14d: 近14天流产率均值
        4. abortion_mean_recent_21d: 近21天流产率均值
        """
        data = self.data.copy()

        data['abortion_mean_recent_7d'] = data.groupby('pigfarm_dk')['abortion_rate']\
            .rolling(window=7, min_periods=7).mean()\
            .reset_index(level=0, drop=True)
        data['abortion_mean_recent_14d'] = data.groupby('pigfarm_dk')['abortion_rate']\
            .rolling(window=14, min_periods=14).mean()\
            .reset_index(level=0, drop=True)
        data['abortion_mean_recent_21d'] = data.groupby('pigfarm_dk')['abortion_rate']\
            .rolling(window=21, min_periods=21).mean()\
            .reset_index(level=0, drop=True)

        self.data = data.copy()

    def _get_boar_feature(self):
        """
        计算种猪类型特征:
        1. boar_transin_times_30d: 30天内猪只转入次数 (boar_transin_qty不为0的天数)
        2. boar_transin_qty_30d: 30天内猪只转入总量
        3. boar_transin_ratio_30d_1: 转入猪只占比1 (基于当前存栏)
        4. boar_transin_ratio_30d_2: 转入猪只占比2 (基于30天量)
        """
        data = self.data.copy()
        # 确保相关列是数值类型
        columns_to_convert = ['boar_transin_qty', 'basesow_sqty', 'basempig_sqty',
                              'reserve_sow_sqty', 'reserve_mpig_sqty']

        for col in columns_to_convert:
            if col in data.columns:
                data[col] = pd.to_numeric(data[col], errors='coerce').fillna(0)
            else:
                logger.warning(f"列 {col} 不存在，将创建并填充为0")
                data[col] = np.nan

        # 1. 计算30天内boar_transin_qty不为0的天数
        # 先创建一个标记，1表示当天有转入，0表示没有
        data['has_boar_transin'] = (data['boar_transin_qty'] > 0).astype(int)

        # 使用rolling计算30天内的转入次数
        data['boar_transin_times_30d'] = data.groupby('pigfarm_dk')['has_boar_transin']\
            .rolling(window=30, min_periods=1).sum()\
            .reset_index(level=0, drop=True)

        # 2. 计算30天内boar_transin_qty的和
        data['boar_transin_qty_30d'] = data.groupby('pigfarm_dk')['boar_transin_qty']\
            .rolling(window=30, min_periods=1).sum()\
            .reset_index(level=0, drop=True)

        # 3. 为第二个比率计算30天其他指标
        data['basesow_sqty_30d_ago'] = data.groupby('pigfarm_dk')['basesow_sqty'].shift(30)

        data['basempig_sqty_30d_ago'] = data.groupby('pigfarm_dk')['basempig_sqty'].shift(30)

        data['reserve_sow_sqty_30d_ago'] = data.groupby('pigfarm_dk')['reserve_sow_sqty'].shift(30)

        data['reserve_mpig_sqty_30d_ago'] = data.groupby('pigfarm_dk')['reserve_mpig_sqty'].shift(30)

        # 4. 计算boar_transin_ratio_30d_1 (基于当前存栏)
        def calculate_boar_ratio(row):
            numerator = row['boar_transin_qty_30d']
            denominator = (row['basesow_sqty'] + row['basempig_sqty'] +
                          row['reserve_sow_sqty'] + row['reserve_mpig_sqty'] +
                          row['boar_transin_qty_30d'])

            if denominator == 0:
                return 0  # 分母为0时返回0
            else:
                return numerator / denominator

        data['boar_transin_ratio_30d_1'] = data.apply(calculate_boar_ratio, axis=1)

        # 5. 计算boar_transin_ratio_30d_2 (基于30天前数量)
        def calculate_boar_ratio_30d(row):
            numerator = row['boar_transin_qty_30d']
            denominator = (row['basesow_sqty_30d_ago'] + row['basempig_sqty_30d_ago'] +
                          row['basempig_sqty_30d_ago'] + row['reserve_sow_sqty_30d_ago'] +
                          row['boar_transin_qty_30d'])

            if denominator == 0:
                return 0  # 分母为0时返回0
            else:
                return numerator / denominator

        data['boar_transin_ratio_30d_2'] = data.apply(calculate_boar_ratio_30d, axis=1)

        self.data = data.copy()
        logger.info("种猪特征计算完成")

    def _get_preg_stock_feature(self):
        """
        计算怀孕母猪特征:
        1. preg_stock_sqty_change_ratio_7d: 与7天前（T-6）相比的存栏量变化率
        2. preg_stock_sqty_change_ratio_15d: 与15天前（T-14）相比的存栏量变化率
        3. preg_stock_sqty: preg_stock_qty对应数据
        """
        data = self.data.copy()
        # 确保preg_stock_qty是数值类型
        data['preg_stock_qty'] = pd.to_numeric(data['preg_stock_qty'], errors='coerce').fillna(0)

        # 按猪场分组并计算6天和14天前的存栏量
        data['preg_stock_qty_6d_ago'] = data.groupby('pigfarm_dk')['preg_stock_qty'].shift(6)
        data['preg_stock_qty_14d_ago'] = data.groupby('pigfarm_dk')['preg_stock_qty'].shift(14)

        # 计算变化率
        def calculate_change_ratio(row, days):
            current = row['preg_stock_qty']
            previous = row[f'preg_stock_qty_{days}d_ago']

            # 处理缺失值
            if pd.isna(previous):
                return np.nan

            # 处理当前值为0的情况
            if current == 0:
                return np.nan

            # 计算变化率
            return (current - previous) / current

        # 应用计算函数
        data['preg_stock_sqty_change_ratio_7d'] = data.apply(
            lambda row: calculate_change_ratio(row, 6), axis=1)

        data['preg_stock_sqty_change_ratio_15d'] = data.apply(
            lambda row: calculate_change_ratio(row, 14), axis=1)

        # 计算后备母猪存栏量
        data['preg_stock_sqty'] = data['preg_stock_qty']

        self.data = data.copy()
        logger.info("怀孕母猪特征计算完成")

    def _get_reserve_sow_feature(self):
        """
        计算后备母猪特征:
        1. reserve_sow_sqty: 后备母猪存栏量
        2. reserve_sow_sqty_change_ratio_7d: 与7天前（T-6）相比的存栏量变化率
        3. reserve_sow_sqty_change_ratio_15d: 与15天前（T-14）相比的存栏量变化率
        """
        data = self.data.copy()
        # 确保reserve_sow_qty是数值类型
        data['reserve_sow_sqty'] = pd.to_numeric(data['reserve_sow_sqty'], errors='coerce').fillna(0)

        # 按猪场分组并计算7天和15天前的存栏量
        data['reserve_sow_sqty_6d_ago'] = data.groupby('pigfarm_dk')['reserve_sow_sqty'].shift(6)
        data['reserve_sow_sqty_14d_ago'] = data.groupby('pigfarm_dk')['reserve_sow_sqty'].shift(14)

        # 修正：按猪场分组计算30天移动平均值
        data['reserve_sow_30day_avg'] = data.groupby('pigfarm_dk')['reserve_sow_sqty']\
            .rolling(window=30, min_periods=30).mean()\
            .reset_index(level=0, drop=True)

        # 计算变化率
        def calculate_change_ratio(row, days):
            current = row['reserve_sow_sqty']
            previous = row[f'reserve_sow_sqty_{days}d_ago']

            # 处理缺失值
            if pd.isna(previous):
                return np.nan

            # 处理当前值为0的情况
            if current == 0:
                return np.nan

            # 计算变化率
            return (current - previous) / current

        # 应用计算函数
        data['reserve_sow_sqty_change_ratio_7d'] = data.apply(
            lambda row: calculate_change_ratio(row, 6), axis=1)

        data['reserve_sow_sqty_change_ratio_15d'] = data.apply(
            lambda row: calculate_change_ratio(row, 14), axis=1)

        # 直接使用reserve_sow_sqty作为reserve_sow_sqty
        data['reserve_sow_sqty'] = data['reserve_sow_sqty']

        self.data = data.copy()
        logger.info("后备母猪特征计算完成")

    def _get_basesow_feature(self):
        """
        计算基础母猪特征:
        1. basesow_sqty: 基础母猪存栏量
        2. basesow_sqty_change_ratio_7d: 与7天前（T-6）相比的存栏量变化率
        3. basesow_sqty_change_ratio_15d: 与15天前（T-14）相比的存栏量变化率
        """
        data = self.data.copy()
        # 确保basesow_qty是数值类型
        data['basesow_sqty'] = pd.to_numeric(data['basesow_sqty'], errors='coerce').fillna(0)

        # 按猪场分组并计算6天和14天前的存栏量
        data['basesow_sqty_6d_ago'] = data.groupby('pigfarm_dk')['basesow_sqty'].shift(6)
        data['basesow_sqty_14d_ago'] = data.groupby('pigfarm_dk')['basesow_sqty'].shift(14)

        # 计算变化率
        def calculate_change_ratio(row, days):
            current = row['basesow_sqty']
            previous = row[f'basesow_sqty_{days}d_ago']

            # 处理缺失值
            if pd.isna(previous):
                return np.nan

            # 处理当前值为0的情况
            if current == 0:
                return None

            # 计算变化率
            return (current - previous) / current

        # 应用计算函数
        data['basesow_sqty_change_ratio_7d'] = data.apply(
            lambda row: calculate_change_ratio(row, 6), axis=1)

        data['basesow_sqty_change_ratio_15d'] = data.apply(
            lambda row: calculate_change_ratio(row, 14), axis=1)

        # 直接使用basesow_qty作为basesow_sqty
        data['basesow_sqty'] = data['basesow_sqty']

        self.data = data.copy()
        logger.info("基础母猪特征计算完成")

    def _get_7d_abortion_feature(self):
        """
        计算近7天流产率特征:
        1. abortion_mean_recent_7d: 近7天流产率均值
        """
        data = self.data.copy()

        # 计算未来1-7天的流产率作为特征
        for day in range(0, 7):
            # 使用shift的负值获取未来数据
            feature_name = f'abortion_rate_past_{day + 1}d'
            data[feature_name] = data.groupby('pigfarm_dk')['abortion_rate'].shift(day)

        self.data = data.copy()
        logger.info("过去7天流产率特征计算完成")

    def _calculate_abortion_rate(self, data, id_column='pigfarm_dk'):
        """
        计算流产率
        """
        # 建立索引
        data.sort_values(by=['stats_dt', id_column], inplace=True)

        # 使用 groupby 和 rolling window 计算每个猪场每个日期的近7天流产总数
        data['recent_7day_abort_sum'] = data.groupby(id_column)['abort_qty'].rolling(window=7, min_periods=7).sum()\
            .reset_index(level=0, drop=True) # reset_index 去掉 groupby 带来的多级索引

        # 定义一个函数来计算流产率，处理分母为0的情况
        def calculate_rate(row):
            sum_recent_abort = row['recent_7day_abort_sum']
            current_preg_stock = row['preg_stock_qty']

            # 如果7天流产总和是NaN (因为窗口不足7天)，则流产率也是NaN
            if pd.isna(sum_recent_abort):
                return np.nan

            # 如果怀孕母猪存栏量是NaN，则流产率也是NaN
            if pd.isna(current_preg_stock):
                return np.nan

            numerator = sum_recent_abort
            denominator = sum_recent_abort + current_preg_stock

            if denominator == 0:
                return np.nan  # 分母为0，也返回 NaN
            else:
                return numerator / denominator

        # 应用计算函数
        data['abortion_rate'] = data.apply(calculate_rate, axis=1)
        # 删除流产率为NaN的行
        data.dropna(subset=['abortion_rate'], inplace=True)

        return data

    def _mark_laner_interval(self, data, threshold=0.0025):
        '''
        标记蓝耳发病区间，流产率由低于阈值到高于阈值当天作为发病期开始，流产率由高于阈值到低于阈值当天且后三天流产率未超过阈值作为发病期结束
        '''
        # 创建副本避免修改原数据
        result_data = data.copy()

        # 初始化is_laner列
        result_data['is_laner'] = 0

        # 按猪场分组处理
        for pigfarm_dk in result_data['pigfarm_dk'].unique():
            # 获取当前猪场的数据
            farm_data = result_data[result_data['pigfarm_dk'] == pigfarm_dk].copy()
            farm_data = farm_data.sort_values('stats_dt').reset_index(drop=True)

            # 标记高于阈值的天数
            farm_data['above_threshold'] = (farm_data['abortion_rate'] > threshold).astype(int)

            # 找到发病区间的开始和结束点
            intervals = []
            in_outbreak = False
            start_idx = None

            for i in range(len(farm_data)):
                current_above = farm_data.iloc[i]['above_threshold']

                # 检查是否为发病开始（从低于阈值到高于阈值）
                if not in_outbreak and current_above == 1:
                    # 检查前一天是否低于阈值（如果存在的话）
                    if i == 0 or farm_data.iloc[i-1]['above_threshold'] == 0:
                        in_outbreak = True
                        start_idx = i

                # 检查是否为发病结束（从高于阈值到低于阈值且连续4天低于阈值）
                elif in_outbreak and current_above == 0:
                    # 检查接下来的3天是否也低于阈值
                    consecutive_below = 1  # 当前天已经低于阈值

                    for j in range(i + 1, min(i + 4, len(farm_data))):
                        if farm_data.iloc[j]['above_threshold'] == 0:
                            consecutive_below += 1
                        else:
                            break

                    # 如果连续4天低于阈值，则认为发病期结束
                    if consecutive_below >= 4:
                        end_idx = i - 1  # 发病结束是高于阈值的最后一天
                        if start_idx is not None:
                            intervals.append((start_idx, end_idx))
                        in_outbreak = False
                        start_idx = None

            # 如果最后还在发病期（数据结束时仍在发病），则以最后一天为结束
            if in_outbreak and start_idx is not None:
                intervals.append((start_idx, len(farm_data) - 1))

            # 标记发病区间
            for start_idx, end_idx in intervals:
                farm_data.loc[start_idx:end_idx, 'is_laner'] = 1

            # 更新到结果数据中
            result_data.loc[result_data['pigfarm_dk'] == pigfarm_dk, 'is_laner'] = farm_data['is_laner'].values

        logger.info(f"标记蓝耳发病区间完成，阈值: {threshold}")
        return result_data

    def _identify_outbreak_periods(self, data):
        """
        识别数据中的发病区间
        返回: [(start_idx, end_idx), ...] 发病区间的索引列表
        """
        outbreak_periods = []

        if 'is_laner' not in data.columns:
            return outbreak_periods

        # 找到连续的发病区间
        in_outbreak = False
        start_idx = None

        for i, row in data.iterrows():
            is_outbreak_day = row['is_laner'] == 1

            if not in_outbreak and is_outbreak_day:
                # 发病开始
                in_outbreak = True
                start_idx = i
            elif in_outbreak and not is_outbreak_day:
                # 发病结束
                if start_idx is not None:
                    outbreak_periods.append((start_idx, i - 1))
                in_outbreak = False
                start_idx = None

        # 如果数据结束时仍在发病期
        if in_outbreak and start_idx is not None:
            outbreak_periods.append((start_idx, len(data) - 1))

        return outbreak_periods

    def _calculate_loss_rate_for_window(self, window_data):
        """
        计算30天窗口内的猪场损失度
        """
        # 确保数据按日期排序
        window_data = window_data.sort_values('stats_dt').reset_index(drop=True)

        # 识别发病区间
        outbreak_periods = self._identify_outbreak_periods(window_data)

        if not outbreak_periods:
            return 0.0  # 没有发病期，损失度为0

        # 计算所有发病时期的abort_qty总和
        total_outbreak_abort_qty = 0
        total_end_preg_stock_qty = 0

        for start_idx, end_idx in outbreak_periods:
            # 发病时期的abort_qty总和
            outbreak_abort_sum = window_data.iloc[start_idx:end_idx+1]['abort_qty'].sum()
            total_outbreak_abort_qty += outbreak_abort_sum

            # 发病末期的preg_stock_qty
            end_preg_stock = window_data.iloc[end_idx]['preg_stock_qty']
            if pd.notna(end_preg_stock):
                total_end_preg_stock_qty += end_preg_stock

        # 计算损失度
        denominator = total_outbreak_abort_qty + total_end_preg_stock_qty

        if denominator == 0:
            return np.nan

        loss_rate = total_outbreak_abort_qty / denominator
        return loss_rate

    def _get_pigfarm_loss_rate(self):
        # 获取流产率
        # 获取流产率并标记蓝耳发病区间
        production_data = self._calculate_abortion_rate(self.production_data.copy(), 'pigfarm_dk')
        production_data = production_data[['stats_dt', 'pigfarm_dk', 'abort_qty', 'preg_stock_qty', 'abortion_rate']].copy()
        production_data = self._mark_laner_interval(production_data)

        # 确保数据按日期排序
        production_data = production_data.sort_values(['pigfarm_dk', 'stats_dt']).reset_index(drop=True)

        # 创建结果数据框
        result_data = []

        # 按猪场分组处理
        for pigfarm_dk in tqdm(production_data['pigfarm_dk'].unique(), desc="计算猪场损失度"):
            farm_data = production_data[production_data['pigfarm_dk'] == pigfarm_dk].copy()
            farm_data = farm_data.sort_values('stats_dt').reset_index(drop=True)

            # 为每个日期计算猪场损失度
            for i, row in farm_data.iterrows():
                current_date = row['stats_dt']
                start_date = current_date - pd.Timedelta(days=29)

                # 获取30天窗口内的数据
                window_data = farm_data[
                    (farm_data['stats_dt'] >= start_date) &
                    (farm_data['stats_dt'] <= current_date)
                ].copy()

                if len(window_data) == 0:
                    result_data.append({
                        'stats_dt': current_date,
                        'pigfarm_dk': pigfarm_dk,
                        'pigfarm_loss_rate': np.nan
                    })
                    continue

                # 计算猪场损失度
                loss_rate = self._calculate_loss_rate_for_window(window_data)

                result_data.append({
                    'stats_dt': current_date,
                    'pigfarm_dk': pigfarm_dk,
                    'pigfarm_loss_rate': loss_rate
                })

        # 转换为DataFrame
        loss_df = pd.DataFrame(result_data)

        # 将猪场损失度合并到主数据中
        if hasattr(self, 'data') and not self.data.empty:
            self.data = self.data.merge(
                loss_df[['stats_dt', 'pigfarm_dk', 'pigfarm_loss_rate']],
                on=['stats_dt', 'pigfarm_dk'],
                how='left'
            )
        else:
            # 如果self.data为空，创建基础数据框
            self.data = loss_df.copy()

        logger.info("猪场损失度计算完成")

    def _calculate_line_loss_rate_for_window(self, window_data):
        """
        计算30天窗口内的生产线损失度
        公式: is_laner==1生产线'abort_qty'和 / (is_laner==1生产线'abort_qty'和 + 每条生产线发病末期'preg_stock_qty'的值)
        """
        # 按生产线分组处理，识别每条生产线的发病区间
        result_data = []
        total_outbreak_abort_qty = 0
        total_end_preg_stock_qty = 0

        # 按生产线分组处理
        for prodline_dk in window_data['prodline_dk'].unique():
            line_data = window_data[window_data['prodline_dk'] == prodline_dk].copy()
            line_data = line_data.sort_values('stats_dt').reset_index(drop=True)

            # 识别该生产线的发病区间
            outbreak_periods = self._identify_outbreak_periods(line_data)

            if outbreak_periods:
                # 计算该生产线所有发病时期的abort_qty总和
                line_outbreak_abort_qty = 0
                for start_idx, end_idx in outbreak_periods:
                    # 发病时期的abort_qty总和
                    outbreak_abort_sum = line_data.iloc[start_idx:end_idx+1]['abort_qty'].sum()
                    line_outbreak_abort_qty += outbreak_abort_sum

                    # 发病末期的preg_stock_qty
                    end_preg_stock = line_data.iloc[end_idx]['preg_stock_qty']
                    if pd.notna(end_preg_stock):
                        total_end_preg_stock_qty += end_preg_stock

                # 累加到总的发病abort_qty中
                total_outbreak_abort_qty += line_outbreak_abort_qty

        # 计算损失度
        denominator = total_outbreak_abort_qty + total_end_preg_stock_qty

        if denominator == 0:
            return 0.0  # 没有发病期或分母为0，损失度为0

        loss_rate = total_outbreak_abort_qty / denominator
        return loss_rate

    def _get_line_loss_rate(self):
        # 获取数据
        production_line_data = self.production_line_data.copy()
        # 将猪场id和生产线id进行拼接
        production_line_data['prodline_dk'] = production_line_data['pigfarm_dk'].astype(str) + '_' + production_line_data['prodline_dk'].astype(str)
        # 计算流产率并标记蓝耳发病区间
        production_line_data = self._calculate_abortion_rate(production_line_data, 'prodline_dk')
        production_line_data = production_line_data[['stats_dt', 'pigfarm_dk', 'prodline_dk', 'abort_qty', 'preg_stock_qty', 'abortion_rate']].copy()
        production_line_data = self._mark_laner_interval(production_line_data, threshold=0.0025)

        result_data = []

        # 按猪场分组处理
        for pigfarm_dk in tqdm(production_line_data['pigfarm_dk'].unique(), desc="计算生产线损失度"):
            farm_data = production_line_data[production_line_data['pigfarm_dk'] == pigfarm_dk].copy()
            farm_data = farm_data.sort_values('stats_dt').reset_index(drop=True)

            # 获取该猪场的所有日期
            dates = farm_data['stats_dt'].unique()

            for current_date in dates:
                start_date = current_date - pd.Timedelta(days=29)

                # 获取30天窗口内的数据
                window_data = farm_data[
                    (farm_data['stats_dt'] >= start_date) &
                    (farm_data['stats_dt'] <= current_date)
                ].copy()

                if len(window_data) == 0:
                    result_data.append({
                        'stats_dt': current_date,
                        'pigfarm_dk': pigfarm_dk,
                        'line_loss_rate': np.nan
                    })
                    continue

                # 计算生产线损失率
                loss_rate = self._calculate_line_loss_rate_for_window(window_data)

                result_data.append({
                    'stats_dt': current_date,
                    'pigfarm_dk': pigfarm_dk,
                    'line_loss_rate': loss_rate
                })

        loss_df = pd.DataFrame(result_data)

        # 将猪场损失度合并到主数据中
        if hasattr(self, 'data') and not self.data.empty:
            self.data = self.data.merge(
                loss_df[['stats_dt', 'pigfarm_dk', 'line_loss_rate']],
                on=['stats_dt', 'pigfarm_dk'],
                how='left'
            )
        else:
            # 如果self.data为空，创建基础数据框
            self.data = loss_df.copy()

        logger.info("猪场损失度计算完成")

    def _calculate_line_health_for_window(self, window_data):
        """
        计算30天窗口内的生产线健康度
        公式: 健康生产线preg_stock_qty / (健康生产线preg_stock_qty + 发病生产线preg_stock_qty + 发病生产线abort_qty)
        """
        # 分离健康和发病的生产线数据
        healthy_lines = window_data[window_data['is_laner'] == 0].copy()
        outbreak_lines = window_data[window_data['is_laner'] == 1].copy()

        if not healthy_lines.empty:
            # 计算健康生产线的preg_stock_qty总和
            healthy_preg_stock = healthy_lines['preg_stock_qty'].iloc[-1]
        else:
            healthy_preg_stock = 0

        if not outbreak_lines.empty:
            # 计算发病生产线的preg_stock_qty总和
            outbreak_preg_stock = outbreak_lines['preg_stock_qty'].iloc[-1]
            # 计算发病生产线的abort_qty总和
            outbreak_abort_sum = outbreak_lines['abort_qty'].sum()
        else:
            outbreak_preg_stock = 0
            outbreak_abort_sum = 0

        # 计算分母
        denominator = healthy_preg_stock + outbreak_preg_stock + outbreak_abort_sum

        # 避免分母为0
        if denominator == 0:
            return np.nan

        # 计算健康度
        health_rate = healthy_preg_stock / denominator

        return health_rate

    def _get_line_health(self):
        """
        计算生产线健康度特征
        公式: 30天内所有is_laner==0的生产线当天的preg_stock_qty /
            (30天内所有is_laner==0的生产线当天的preg_stock_qty +
            30天内存在is_laner==1的生产线当天的preg_stock_qty +
            30天内存在is_laner==1的生产线的abort_qty的和)
        """
        # 获取数据
        production_line_data = self.production_line_data.copy()

        # 确保必要的列存在并转换数据类型
        required_columns = ['stats_dt', 'pigfarm_dk', 'prodline_dk', 'abort_qty', 'preg_stock_qty']
        for col in required_columns:
            if col not in production_line_data.columns:
                logger.error(f"缺少必要列: {col}")
                return

        # 转换数据类型
        production_line_data['abort_qty'] = pd.to_numeric(production_line_data['abort_qty'], errors='coerce').fillna(0)
        production_line_data['preg_stock_qty'] = pd.to_numeric(production_line_data['preg_stock_qty'], errors='coerce').fillna(0)

        # 将猪场id和生产线id进行拼接
        production_line_data['prodline_dk'] = production_line_data['pigfarm_dk'].astype(str) + '_' + production_line_data['prodline_dk'].astype(str)

        # 计算流产率并标记蓝耳发病区间
        production_line_data = self._calculate_abortion_rate(production_line_data, 'prodline_dk')
        production_line_data = production_line_data[['stats_dt', 'pigfarm_dk', 'prodline_dk', 'abort_qty', 'preg_stock_qty', 'abortion_rate']].copy()
        production_line_data = self._mark_laner_interval(production_line_data, threshold=0.0025)

        result_data = []

        # 按猪场分组处理
        for pigfarm_dk in tqdm(production_line_data['pigfarm_dk'].unique(), desc="计算生产线健康度"):
            farm_data = production_line_data[production_line_data['pigfarm_dk'] == pigfarm_dk].copy()
            farm_data = farm_data.sort_values('stats_dt').reset_index(drop=True)

            # 获取该猪场的所有日期
            dates = farm_data['stats_dt'].unique()

            for current_date in dates:
                start_date = current_date - pd.Timedelta(days=29)

                # 获取30天窗口内的数据
                window_data = farm_data[
                    (farm_data['stats_dt'] >= start_date) &
                    (farm_data['stats_dt'] <= current_date)
                ].copy()

                if len(window_data) == 0:
                    result_data.append({
                        'stats_dt': current_date,
                        'pigfarm_dk': pigfarm_dk,
                        'line_health_rate': np.nan
                    })
                    continue

                # 计算生产线健康度
                health_rate = self._calculate_line_health_for_window(window_data)

                result_data.append({
                    'stats_dt': current_date,
                    'pigfarm_dk': pigfarm_dk,
                    'line_health_rate': health_rate
                })

        # 转换为DataFrame
        health_df = pd.DataFrame(result_data)

        # 将生产线健康度合并到主数据中
        if hasattr(self, 'data') and not self.data.empty:
            self.data = self.data.merge(
                health_df[['stats_dt', 'pigfarm_dk', 'line_health_rate']],
                on=['stats_dt', 'pigfarm_dk'],
                how='left'
            )
        else:
            # 如果self.data为空，创建基础数据框
            self.data = health_df.copy()

        logger.info("生产线健康度计算完成")

    def _post_processing_data(self):
        data = self.data.copy()
        if data.isnull().any().any():
            logger.info("Warning: Null in production_feature_data.csv")
        self.file_name = "production_feature_data." + self.file_type

        past_7d_abortion_features = [f'abortion_rate_past_{day + 1}d' for day in range(7)]
        production_feature_list = ['stats_dt', 'pigfarm_dk', 'abortion_rate','abortion_mean_recent_7d',
                                   'abortion_mean_recent_14d', 'abortion_mean_recent_21d',
                                   'boar_transin_times_30d', 'boar_transin_qty_30d',
                                   'boar_transin_ratio_30d_1', 'boar_transin_ratio_30d_2',
                                   'preg_stock_sqty_change_ratio_7d', 'preg_stock_sqty_change_ratio_15d','preg_stock_sqty',
                                   'reserve_sow_sqty_change_ratio_7d', 'reserve_sow_sqty_change_ratio_15d','reserve_sow_sqty', 'reserve_sow_30day_avg',
                                   'basesow_sqty_change_ratio_7d', 'basesow_sqty_change_ratio_15d','basesow_sqty',
                                   'pigfarm_loss_rate', 'line_loss_rate', 'line_health_rate'] + past_7d_abortion_features

        data = data[production_feature_list]
        data['stats_dt'] = pd.to_datetime(data['stats_dt'])
        data['stats_dt'] = data['stats_dt'] + pd.DateOffset(days=1)
        self.data = data.copy()

    def build_dataset_all(self):
        logger.info("-----Preprocessing data----- ")
        self._preprocessing_data()
        logger.info("Calculating interval from last purchase...")
        logger.info("-----get abortion rate-----")
        self._get_abortion_rate()
        logger.info("-----get abortion mean feature-----")
        self._get_abortion_mean_feature()
        logger.info("-----get boar feature-----")
        self._get_boar_feature()
        logger.info("-----get preg stock feature-----")
        self._get_preg_stock_feature()
        logger.info("-----get reserve sow feature-----")
        self._get_reserve_sow_feature()
        logger.info("-----get basesow feature-----")
        self._get_basesow_feature()
        logger.info("-----get past abortion feature-----")
        self._get_7d_abortion_feature()
        logger.info("-----get pigfarm loss rate-----")
        self._get_pigfarm_loss_rate()
        logger.info("-----get line loss rate-----")
        self._get_line_loss_rate()
        logger.info("-----get line health rate-----")
        self._get_line_health()
        logger.info("-----Postprocessing data----- ")
        self._post_processing_data()
        # logger.info("-----Save as : {}".format("/".join([config.FEATURE_STORE_ROOT, self.file_name])))
        logger.info("-----Save as : {}".format(config.FeatureData.PRODUCTION_FEATURE_DATA.value))
        # self.dump_dataset("/".join([config.FEATURE_STORE_ROOT, self.file_name]))
        self.dump_dataset(config.FeatureData.PRODUCTION_FEATURE_DATA.value)
        logger.info("-----Dataset saved successfully-----")

if __name__ == "__main__":
    # Example usage
    running_dt_end = "2024-12-31"
    train_interval = 30
    file_type = "csv"

    dataset = ProductionFeature(running_dt_end, train_interval, file_type)
    dataset.build_dataset_all()
    logger.info("Production feature dataset built successfully.")



