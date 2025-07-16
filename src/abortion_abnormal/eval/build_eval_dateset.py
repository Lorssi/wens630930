import sys
from pathlib import Path
import pandas as pd
import numpy as np
from tqdm import tqdm

# 获取项目根目录（假设是 src 的父目录）
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_root))

from configs.feature_config import DataPathConfig

def calculate_abortion_rate(eval_running_dt_start=None, eval_running_dt_end=None):
    """
    计算流产率
    """
    # 读取原始数据
    production_data = pd.read_csv(DataPathConfig.ML_DATA_PATH, encoding='utf-8')
    # 筛选日期范围，加快计算速度
    production_data['stats_dt'] = pd.to_datetime(production_data['stats_dt'])
    production_data = production_data[
        (production_data['stats_dt'] >= pd.to_datetime(eval_running_dt_start) - pd.Timedelta(days=30)) &
        (production_data['stats_dt'] <= pd.to_datetime(eval_running_dt_end) + pd.Timedelta(days=30))
    ]
    # 建立索引
    production_data.sort_values(by=['stats_dt', 'pigfarm_dk'], inplace=True)

    # 计算流产率

    # 使用 groupby 和 rolling window 计算每个猪场每个日期的近7天流产总数
    production_data['recent_7day_abort_sum'] = production_data.groupby('pigfarm_dk')['abort_qty'].rolling(window=7, min_periods=7).sum()\
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
    production_data['abortion_rate'] = production_data.apply(calculate_rate, axis=1)
    # 删除流产率为NaN的行
    production_data.dropna(subset=['abortion_rate'], inplace=True)

    return production_data[['stats_dt', 'pigfarm_dk', 'abortion_rate']].copy()


def mark_abnormal_abortion_rates(abortion_rate_data, threshold=0.0025):
    """
    标记受异常流产率影响的样本

    参数:
        abortion_rate_data: 包含stats_dt, pigfarm_dk, abortion_rate的DataFrame
        threshold: 流产率阈值

    返回:
        添加了单次异常影响标记的DataFrame
    """
    # 创建结果DataFrame的副本
    result = abortion_rate_data.copy()

    # 初始化影响标记列为0
    result['single_influence_1_7'] = 0
    result['single_influence_8_14'] = 0
    result['single_influence_15_21'] = 0

    # 为每个猪场处理
    for pigfarm, farm_data in tqdm(result.groupby('pigfarm_dk'), desc="为每个猪场标记异常流产率"):
        # 查找异常流产率的日期
        farm_data_sorted = farm_data.sort_values('stats_dt')

        # 找出异常流产率的日期（满足条件：t天流产率>阈值，t-1和t+1天流产率<阈值）
        for i in range(1, len(farm_data_sorted) - 1):
            prev_row = farm_data_sorted.iloc[i-1]
            curr_row = farm_data_sorted.iloc[i]
            next_row = farm_data_sorted.iloc[i+1]

            # 检查是否符合异常流产率的条件
            if (curr_row['abortion_rate'] > threshold and
                prev_row['abortion_rate'] <= threshold and
                next_row['abortion_rate'] <= threshold):

                abnormal_date = curr_row['stats_dt']

                # 计算需要标记的三个时间窗口的日期
                influence_1_7_dates = [abnormal_date - pd.Timedelta(days=d) for d in [9, 8, 7, 6]]
                influence_8_14_dates = [abnormal_date - pd.Timedelta(days=d) for d in [16, 15, 14, 13]]
                influence_15_21_dates = [abnormal_date - pd.Timedelta(days=d) for d in [23, 22, 21, 20]]

                # 标记第一个窗口 (1-7天)
                mask_1_7 = (result['pigfarm_dk'] == pigfarm) & (result['stats_dt'].isin(influence_1_7_dates))
                result.loc[mask_1_7, 'single_influence_1_7'] = 1

                # 标记第二个窗口 (8-14天)
                mask_8_14 = (result['pigfarm_dk'] == pigfarm) & (result['stats_dt'].isin(influence_8_14_dates))
                result.loc[mask_8_14, 'single_influence_8_14'] = 1

                # 标记第三个窗口 (15-21天)
                mask_15_21 = (result['pigfarm_dk'] == pigfarm) & (result['stats_dt'].isin(influence_15_21_dates))
                result.loc[mask_15_21, 'single_influence_15_21'] = 1

    return result


def abortion_abnormal_index_sample(eval_running_dt_start=None, eval_running_dt_end=None, threshold=0.0025, use_cache=False):
    """
    生成任务1：猪场异常预警的index_sample和index_ground_truth
    """
    if use_cache:
        index_sample = pd.read_csv(r'data\interim_data\eval\abortion_abnormal_eval_data\abortion_abnormal_index_sample.csv')
        index_sample['stats_dt'] = pd.to_datetime(index_sample['stats_dt'])
        ground_truth_index = pd.read_csv(r'data\interim_data\eval\abortion_abnormal_eval_data\abortion_abnormal_ground_truth.csv')
        ground_truth_index['stats_dt'] = pd.to_datetime(ground_truth_index['stats_dt'])
        return index_sample, ground_truth_index

    if eval_running_dt_start is None or eval_running_dt_end is None:
        raise ValueError("评估期的开始和结束日期不能为空")

    # 计算流产率
    abortion_rate_data = calculate_abortion_rate(eval_running_dt_start, eval_running_dt_end)

    # 标记异常流产率
    abortion_rate_data = mark_abnormal_abortion_rates(abortion_rate_data, threshold)

    # 筛选测试集范围内的数据
    eval_start = pd.to_datetime(eval_running_dt_start)
    eval_end = pd.to_datetime(eval_running_dt_end)
    # 需要包含评估期之后的数据，以计算未来的标签
    extended_end = eval_end + pd.Timedelta(days=22)
    ground_truth_index = abortion_rate_data[(abortion_rate_data['stats_dt'] >= eval_start) &
                                (abortion_rate_data['stats_dt'] <= extended_end)].copy()

    # 生成真实标签，为每个日期和猪场生成未来三个时间窗口的标签
    # 为每个猪场处理
    for pigfarm, farm_data in tqdm(ground_truth_index.groupby('pigfarm_dk'), desc="为每个猪场生成真实标签"):
        # 获取排序后的数据
        farm_data_sorted = farm_data.sort_values('stats_dt')

        for idx, row in farm_data_sorted.iterrows():
            current_dt = row['stats_dt']

            # 计算未来1-7天的标签
            future_1_7 = farm_data_sorted[
                (farm_data_sorted['stats_dt'] > current_dt) &
                (farm_data_sorted['stats_dt'] <= current_dt + pd.Timedelta(days=7))
            ]
            if any(future_1_7['abortion_rate'] > threshold):
                ground_truth_index.loc[idx, 'abort_1_7'] = 1
            else:
                if len(future_1_7) != 7:
                    continue
                ground_truth_index.loc[idx, 'abort_1_7'] = 0

            # 计算未来8-14天的标签
            future_8_14 = farm_data_sorted[
                (farm_data_sorted['stats_dt'] > current_dt + pd.Timedelta(days=7)) &
                (farm_data_sorted['stats_dt'] <= current_dt + pd.Timedelta(days=14))
            ]
            if any(future_8_14['abortion_rate'] > threshold):
                ground_truth_index.loc[idx, 'abort_8_14'] = 1
            else:
                if len(future_8_14) != 7:
                    continue
                ground_truth_index.loc[idx, 'abort_8_14'] = 0

            # 计算未来15-21天的标签
            future_15_21 = farm_data_sorted[
                (farm_data_sorted['stats_dt'] > current_dt + pd.Timedelta(days=14)) &
                (farm_data_sorted['stats_dt'] <= current_dt + pd.Timedelta(days=21))
            ]
            if any(future_15_21['abortion_rate'] > threshold):
                ground_truth_index.loc[idx, 'abort_15_21'] = 1
            else:
                if len(future_15_21) != 7:
                    continue
                ground_truth_index.loc[idx, 'abort_15_21'] = 0

    # 只保留评估期内的数据
    ground_truth_index = ground_truth_index[
        (ground_truth_index['stats_dt'] >= eval_start) &
        (ground_truth_index['stats_dt'] <= eval_end)
    ]
    # 删除空值的行
    ground_truth_index.dropna(subset=['abort_1_7', 'abort_8_14', 'abort_15_21'], inplace=True)

    # 创建index_ground_truth
    index_ground_truth = ground_truth_index[['stats_dt', 'pigfarm_dk', 'abort_1_7', 'abort_8_14', 'abort_15_21', 'single_influence_1_7', 'single_influence_8_14', 'single_influence_15_21']].copy()

    # 创建index_sample
    index_sample = ground_truth_index[['stats_dt', 'pigfarm_dk']].copy()

    return index_sample, index_ground_truth


def abortion_abnormal_index_sample_v2(eval_running_dt_start=None, eval_running_dt_end=None, threshold=0.0025, use_cache=False):
    """
    生成任务1：猪场异常预警的index_sample和index_ground_truth v2版本
    """
    if use_cache:
        index_sample = pd.read_csv(r'data\interim_data\eval\abortion_abnormal_eval_data\abortion_abnormal_index_sample.csv')
        index_sample['stats_dt'] = pd.to_datetime(index_sample['stats_dt'])
        ground_truth_index = pd.read_csv(r'data\interim_data\eval\abortion_abnormal_eval_data\abortion_abnormal_ground_truth.csv')
        ground_truth_index['stats_dt'] = pd.to_datetime(ground_truth_index['stats_dt'])
        return index_sample, ground_truth_index

    if eval_running_dt_start is None or eval_running_dt_end is None:
        raise ValueError("评估期的开始和结束日期不能为空")

    # 计算流产率
    abortion_rate_data = calculate_abortion_rate(eval_running_dt_start, eval_running_dt_end)

    # 标记异常流产率
    abortion_rate_data = mark_abnormal_abortion_rates(abortion_rate_data, threshold)

    # 筛选测试集范围内的数据
    eval_start = pd.to_datetime(eval_running_dt_start)
    eval_end = pd.to_datetime(eval_running_dt_end)
    # 需要包含评估期之后的数据，以计算未来的标签
    extended_start = eval_start - pd.Timedelta(days=30)
    extended_end = eval_end + pd.Timedelta(days=30)
    ground_truth_index = abortion_rate_data[(abortion_rate_data['stats_dt'] >= extended_start) &
                                (abortion_rate_data['stats_dt'] <= extended_end)].copy()

    # 生成真实标签，为每个日期和猪场生成未来三个时间窗口的标签
    # 为每个猪场处理
    for pigfarm, farm_data in tqdm(ground_truth_index.groupby('pigfarm_dk'), desc="为每个猪场生成真实标签"):
        # 获取排序后的数据
        farm_data_sorted = farm_data.sort_values('stats_dt')

        for idx, row in farm_data_sorted.iterrows():
            current_dt = row['stats_dt']

            # 计算未来1-7天的标签
            future_1_7 = farm_data_sorted[
                (farm_data_sorted['stats_dt'] >= current_dt - pd.Timedelta(days=6)) &
                (farm_data_sorted['stats_dt'] <= current_dt + pd.Timedelta(days=7))
            ]
            if any(future_1_7['abortion_rate'] > threshold):
                ground_truth_index.loc[idx, 'abort_1_7'] = 1
            else:
                if len(future_1_7) != 14:
                    continue
                ground_truth_index.loc[idx, 'abort_1_7'] = 0

            # 计算未来8-14天的标签
            future_8_14 = farm_data_sorted[
                (farm_data_sorted['stats_dt'] >= current_dt + pd.Timedelta(days=1)) &
                (farm_data_sorted['stats_dt'] <= current_dt + pd.Timedelta(days=14))
            ]
            if any(future_8_14['abortion_rate'] > threshold):
                ground_truth_index.loc[idx, 'abort_8_14'] = 1
            else:
                if len(future_8_14) != 14:
                    continue
                ground_truth_index.loc[idx, 'abort_8_14'] = 0

            # 计算未来15-21天的标签
            future_15_21 = farm_data_sorted[
                (farm_data_sorted['stats_dt'] >= current_dt + pd.Timedelta(days=8)) &
                (farm_data_sorted['stats_dt'] <= current_dt + pd.Timedelta(days=21))
            ]
            if any(future_15_21['abortion_rate'] > threshold):
                ground_truth_index.loc[idx, 'abort_15_21'] = 1
            else:
                if len(future_15_21) != 14:
                    continue
                ground_truth_index.loc[idx, 'abort_15_21'] = 0

    # 只保留评估期内的数据
    ground_truth_index = ground_truth_index[
        (ground_truth_index['stats_dt'] >= eval_start) &
        (ground_truth_index['stats_dt'] <= eval_end)
    ]
    # 删除空值的行
    ground_truth_index.dropna(subset=['abort_1_7', 'abort_8_14', 'abort_15_21'], inplace=True)

    # 创建index_ground_truth
    index_ground_truth = ground_truth_index[['stats_dt', 'pigfarm_dk', 'abort_1_7', 'abort_8_14', 'abort_15_21', 'single_influence_1_7', 'single_influence_8_14', 'single_influence_15_21']].copy()

    # 创建index_sample
    index_sample = ground_truth_index[['stats_dt', 'pigfarm_dk']].copy()

    return index_sample, index_ground_truth


def abortion_abnormal_index_sample_v3(eval_running_dt_start=None, eval_running_dt_end=None, threshold=0.0025, use_cache=False):
    """
    生成任务1：猪场异常预警的index_sample和index_ground_truth
    """
    if use_cache:
        index_sample = pd.read_csv(r'data\interim_data\eval\abortion_abnormal_eval_data\abortion_abnormal_index_sample.csv')
        index_sample['stats_dt'] = pd.to_datetime(index_sample['stats_dt'])
        ground_truth_index = pd.read_csv(r'data\interim_data\eval\abortion_abnormal_eval_data\abortion_abnormal_ground_truth.csv')
        ground_truth_index['stats_dt'] = pd.to_datetime(ground_truth_index['stats_dt'])
        return index_sample, ground_truth_index

    if eval_running_dt_start is None or eval_running_dt_end is None:
        raise ValueError("评估期的开始和结束日期不能为空")

    # 计算流产率
    abortion_rate_data = calculate_abortion_rate(eval_running_dt_start, eval_running_dt_end)

    # 标记异常流产率
    abortion_rate_data = mark_abnormal_abortion_rates(abortion_rate_data, threshold)

    # 筛选测试集范围内的数据
    eval_start = pd.to_datetime(eval_running_dt_start)
    eval_end = pd.to_datetime(eval_running_dt_end)
    # 需要包含评估期之后的数据，以计算未来的标签
    extended_end = eval_end + pd.Timedelta(days=22)
    ground_truth_index = abortion_rate_data[(abortion_rate_data['stats_dt'] >= eval_start) &
                                (abortion_rate_data['stats_dt'] <= extended_end)].copy()

    # 生成真实标签，为每个日期和猪场生成未来三个时间窗口的标签
    # 为每个猪场处理
    for pigfarm, farm_data in tqdm(ground_truth_index.groupby('pigfarm_dk'), desc="为每个猪场生成真实标签"):
        # 获取排序后的数据
        farm_data_sorted = farm_data.sort_values('stats_dt')

        for idx, row in farm_data_sorted.iterrows():
            current_dt = row['stats_dt']

            # 计算未来1-7天的标签
            future_1_7 = farm_data_sorted[
                (farm_data_sorted['stats_dt'] > current_dt) &
                (farm_data_sorted['stats_dt'] <= current_dt + pd.Timedelta(days=7))
            ]
            if any(future_1_7['abortion_rate'] > threshold):
                ground_truth_index.loc[idx, 'abort_1_7'] = 1
            else:
                if len(future_1_7) != 7:
                    continue
                ground_truth_index.loc[idx, 'abort_1_7'] = 0

            # 计算未来8-14天的标签
            future_8_14 = farm_data_sorted[
                (farm_data_sorted['stats_dt'] > current_dt + pd.Timedelta(days=7)) &
                (farm_data_sorted['stats_dt'] <= current_dt + pd.Timedelta(days=14))
            ]
            if any(future_8_14['abortion_rate'] > threshold):
                ground_truth_index.loc[idx, 'abort_8_14'] = 1
            else:
                if len(future_8_14) != 7:
                    continue
                ground_truth_index.loc[idx, 'abort_8_14'] = 0

            # 计算未来15-21天的标签
            future_15_21 = farm_data_sorted[
                (farm_data_sorted['stats_dt'] > current_dt + pd.Timedelta(days=14)) &
                (farm_data_sorted['stats_dt'] <= current_dt + pd.Timedelta(days=21))
            ]
            if any(future_15_21['abortion_rate'] > threshold):
                ground_truth_index.loc[idx, 'abort_15_21'] = 1
            else:
                if len(future_15_21) != 7:
                    continue
                ground_truth_index.loc[idx, 'abort_15_21'] = 0

    # 只保留评估期内的数据
    ground_truth_index = ground_truth_index[
        (ground_truth_index['stats_dt'] >= eval_start) &
        (ground_truth_index['stats_dt'] <= eval_end)
    ]
    # 删除空值的行
    ground_truth_index.dropna(subset=['abort_1_7', 'abort_8_14', 'abort_15_21'], inplace=True)

    # 新口径，如果遇到01，将前面4天均设为1
    for pigfarm, farm_data in tqdm(ground_truth_index.groupby('pigfarm_dk'), desc="应用新口径规则"):
        # 获取排序后的数据
        farm_data_sorted = farm_data.sort_values('stats_dt').reset_index(drop=True)

        # 为每个时间周期处理
        for period in ['abort_1_7', 'abort_8_14', 'abort_15_21']:
            for i in range(len(farm_data_sorted) - 1):
                current_row = farm_data_sorted.iloc[i]
                next_row = farm_data_sorted.iloc[i + 1]

                # 检查是否符合条件：第t天为0，第t+1天为1
                if (current_row[period] == 0 and next_row[period] == 1):
                    current_date = current_row['stats_dt']

                    # 找到需要修改的日期范围（t-3, t-2, t-1, t）
                    target_dates = [
                        current_date - pd.Timedelta(days=3),
                        current_date - pd.Timedelta(days=2),
                        current_date - pd.Timedelta(days=1),
                        current_date
                    ]

                    # 更新这些日期的标签为1
                    for target_date in target_dates:
                        mask = (ground_truth_index['pigfarm_dk'] == pigfarm) & (ground_truth_index['stats_dt'] == target_date)
                        if mask.any():
                            ground_truth_index.loc[mask, period] = 1

    # 创建index_ground_truth
    index_ground_truth = ground_truth_index[['stats_dt', 'pigfarm_dk', 'abort_1_7', 'abort_8_14', 'abort_15_21']].copy()

    # 创建index_sample
    index_sample = ground_truth_index[['stats_dt', 'pigfarm_dk']].copy()

    return index_sample, index_ground_truth

def abortion_days_index_sample(eval_running_dt_start=None, eval_running_dt_end=None, threshold=0.0025):
    """
    生成任务2：流产天数的index_sample和index_ground_truth
    """
    if eval_running_dt_start is None or eval_running_dt_end is None:
        raise ValueError("评估期的开始和结束日期不能为空")

    # 计算流产率
    abortion_rate_data = calculate_abortion_rate(eval_running_dt_start, eval_running_dt_end)

    # 标记流产率异常导致的影响
    abortion_rate_data = mark_abnormal_abortion_rates(abortion_rate_data, threshold)

    # 筛选测试集范围内的数据
    eval_start = pd.to_datetime(eval_running_dt_start)
    eval_end = pd.to_datetime(eval_running_dt_end)
    # 需要包含评估期之后的数据，以计算未来的标签
    extended_end = eval_end + pd.Timedelta(days=22)
    ground_truth_index = abortion_rate_data[(abortion_rate_data['stats_dt'] >= eval_start) &
                                (abortion_rate_data['stats_dt'] <= extended_end)].copy()
    ground_truth_index = ground_truth_index.sort_values(by=['pigfarm_dk', 'stats_dt'])

    # 生成真实标签，为每个日期和猪场生成未来三个时间窗口的标签
    # 为每个猪场处理
    for pigfarm, farm_data in tqdm(ground_truth_index.groupby('pigfarm_dk'), desc="为每个猪场生成真实标签"):
        # 获取排序后的数据
        farm_data_sorted = farm_data.sort_values('stats_dt')
        for idx, row in farm_data_sorted.iterrows():
            current_dt = row['stats_dt']
            # 计算未来1-7天的标签
            future_1_7 = farm_data_sorted[
                (farm_data_sorted['stats_dt'] > current_dt) &
                (farm_data_sorted['stats_dt'] <= current_dt + pd.Timedelta(days=7))
            ]
            if len(future_1_7) == 7:
                ground_truth_index.loc[idx, 'abort_days_1_7'] = sum(future_1_7['abortion_rate'] > threshold)

            # 计算未来8-14天的标签
            future_8_14 = farm_data_sorted[
                (farm_data_sorted['stats_dt'] > current_dt + pd.Timedelta(days=7)) &
                (farm_data_sorted['stats_dt'] <= current_dt + pd.Timedelta(days=14))
            ]
            if len(future_8_14) == 7:
                ground_truth_index.loc[idx, 'abort_days_8_14'] = sum(future_8_14['abortion_rate'] > threshold)

            # 计算未来15-21天的标签
            future_15_21 = farm_data_sorted[
                (farm_data_sorted['stats_dt'] > current_dt + pd.Timedelta(days=14)) &
                (farm_data_sorted['stats_dt'] <= current_dt + pd.Timedelta(days=21))
            ]
            if len(future_15_21) == 7:
                ground_truth_index.loc[idx, 'abort_days_15_21'] = sum(future_15_21['abortion_rate'] > threshold)

    # 只保留评估期内的数据
    ground_truth_index = ground_truth_index[
        (ground_truth_index['stats_dt'] >= eval_start) &
        (ground_truth_index['stats_dt'] <= eval_end)
    ]

    # 删除空值的行
    ground_truth_index.dropna(subset=['abort_days_1_7', 'abort_days_8_14', 'abort_days_15_21'], inplace=True)

    # 创建index_ground_truth
    index_ground_truth = ground_truth_index[['stats_dt', 'pigfarm_dk',
                                             'abort_days_1_7', 'abort_days_8_14', 'abort_days_15_21',
                                             'single_influence_1_7', 'single_influence_8_14', 'single_influence_15_21']].copy()

    # 创建index_sample
    index_sample = ground_truth_index[['stats_dt', 'pigfarm_dk']].copy()

    return index_sample, index_ground_truth



if __name__ == "__main__":
    for start_date, end_date in [
        ('2024-03-01', '2024-04-30'),
        ('2024-06-01', '2024-07-31'),
        ('2024-09-01', '2024-10-31'),
        ('2024-12-01', '2025-01-31'),
    ]:
        # index_sample_1, index_ground_truth_1 = abortion_abnormal_index_sample(start_date, end_date)
        # index_sample_2, index_ground_truth_2 = abortion_abnormal_index_sample_v2(start_date, end_date)
        index_sample_3, index_ground_truth_3 = abortion_abnormal_index_sample_v3(start_date, end_date)
        index_sample_3.to_csv(f'index_sample_v3_{start_date}.csv', index=False, encoding='utf-8')
        # index_ground_truth.to_csv(f'v1_{start_date}.csv', index=False, encoding='utf-8')

