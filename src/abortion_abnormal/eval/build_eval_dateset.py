import pandas as pd
import numpy as np

from configs.feature_config import DataPathConfig


import pandas as pd
import numpy as np

from configs.feature_config import DataPathConfig


def abortion_abnormal_index_sample(eval_running_dt_start=None, eval_running_dt_end=None):
    """
    生成任务1：猪场异常预警的index_sample和index_ground_truth
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
    # 确保流产数量和怀孕母猪存栏量是数值类型，并将NaN填充为0，因为它们参与计算
    production_data['abort_qty'] = pd.to_numeric(production_data['abort_qty'], errors='coerce').fillna(0)
    production_data['preg_stock_qty'] = pd.to_numeric(production_data['preg_stock_qty'], errors='coerce').fillna(0)

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
    production_data = production_data.dropna(subset=['abortion_rate'])

    # 筛选测试集范围内的数据
    if eval_running_dt_start is not None and eval_running_dt_end is not None:
        eval_start = pd.to_datetime(eval_running_dt_start)
        eval_end = pd.to_datetime(eval_running_dt_end)
        # 需要包含评估期之后的数据，以计算未来的标签
        extended_end = eval_end + pd.Timedelta(days=22)
        eval_data = production_data[(production_data['stats_dt'] >= eval_start) & 
                                  (production_data['stats_dt'] <= extended_end)]
    else:
        eval_data = production_data
    
    # 生成真实标签
    def generate_ground_truth(df, threshold=0.0025):
        """
        为每个日期和猪场生成未来三个时间窗口的标签
        """
        # 初始化标签列
        df = df.copy()
        df['abort_1_7'] = 0
        df['abort_8_14'] = 0
        df['abort_15_21'] = 0
        
        # 为每个猪场处理
        for pigfarm, farm_data in df.groupby('pigfarm_dk'):
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
                    df.loc[idx, 'abort_1_7'] = 1
                
                # 计算未来8-14天的标签
                future_8_14 = farm_data_sorted[
                    (farm_data_sorted['stats_dt'] > current_dt + pd.Timedelta(days=7)) & 
                    (farm_data_sorted['stats_dt'] <= current_dt + pd.Timedelta(days=14))
                ]
                if any(future_8_14['abortion_rate'] > threshold):
                    df.loc[idx, 'abort_8_14'] = 1
                
                # 计算未来15-21天的标签
                future_15_21 = farm_data_sorted[
                    (farm_data_sorted['stats_dt'] > current_dt + pd.Timedelta(days=14)) & 
                    (farm_data_sorted['stats_dt'] <= current_dt + pd.Timedelta(days=21))
                ]
                if any(future_15_21['abortion_rate'] > threshold):
                    df.loc[idx, 'abort_15_21'] = 1
        
        return df

    ground_truth_data = generate_ground_truth(eval_data)

    # 只保留评估期内的数据
    if eval_running_dt_start is not None and eval_running_dt_end is not None:
        ground_truth_data = ground_truth_data[
            (ground_truth_data['stats_dt'] >= eval_start) & 
            (ground_truth_data['stats_dt'] <= eval_end)
        ]

    # 创建index_ground_truth
    index_ground_truth = ground_truth_data[['stats_dt', 'pigfarm_dk', 'abort_1_7', 'abort_8_14', 'abort_15_21']].copy()

    # 创建index_sample
    index_sample = ground_truth_data[['stats_dt', 'pigfarm_dk']].copy()

    return index_sample, index_ground_truth
