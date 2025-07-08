import sys
import os
from pathlib import Path
import pandas as pd
from tqdm import tqdm

from abortion_abnormal.eval.build_eval_dateset import abortion_abnormal_index_sample_v2

# ads_pig_efficient_piglet_batch_analysis_day.csv 统计积压仔猪
#     org_farm_dk：猪场数据键
#     pd03010103：仔猪天龄
#     stats_dt：统计日期
#     pd25010316：猪苗数量

# ADS_PIG_ISOLATION_TAME_PROLINE_RISK.csv 引种入群检测
#     prorg_inv_dk：猪场数据键
#     allot_dt: 引种日期
#     min_boar_inpop_dt：入群日期
#     yzbe8_blue_ear_kyyd_check_out_qty：后备引种前8-30天蓝耳野毒检出数
#     rqbe3_blue_ear_kypt_check_out_qty：后备入群前3天蓝耳普通蓝耳检出数
#     rqbe3_blue_ear_kyyd_check_out_qty：后备入群前3天蓝耳野毒检出数

# index_sample
#     stats_dt: 统计日期
#     pigfarm_dk: 猪场数据键

# 1. 入群前3天猪只检出抗原阳性
# 2. 超100头猪苗积压，且猪苗天龄大于40
# 3. 引种前8-30天猪只蓝耳（野毒）检出阳性

class RuleBaseline:

    def __init__(self, running_start_date, running_end_date, index_sample):
        self.running_start_date = pd.to_datetime(running_start_date)
        self.running_end_date = pd.to_datetime(running_end_date)
        self.index_sample = index_sample
        self.get_data()

        # delete 添加abort_1_7_decision，为了使用之前的评测代码
        index_sample['abort_1_7_decision'] = 0


    def get_data(self):
        # 获取数据
        self.piglets_data = pd.read_csv("data/raw_data/ads_pig_efficient_piglet_batch_analysis_day.csv", encoding='utf-8')
        self.piglets_data['stats_dt'] = pd.to_datetime(self.piglets_data['stats_dt'], format='mixed')
        self.piglets_data = self.piglets_data[(self.piglets_data['stats_dt'] >= self.running_start_date - pd.Timedelta(days=30)) &
                                              (self.piglets_data['stats_dt'] <= self.running_end_date)]

        self.intro_tame_data = pd.read_csv("data/raw_data/ads_pig_isolation_tame_proline_risk.csv", encoding='utf-8', low_memory=False)
        self.intro_tame_data['allot_dt'] = pd.to_datetime(self.intro_tame_data['allot_dt'])
        self.intro_tame_data['min_boar_inpop_dt'] = pd.to_datetime(self.intro_tame_data['min_boar_inpop_dt'])

        self.index_sample['stats_dt'] = pd.to_datetime(self.index_sample['stats_dt'])
        self.index_sample.sort_values(by=['pigfarm_dk', 'stats_dt'], inplace=True)


    def judge_contition(self, stats_dt, day, data, column):
        start = stats_dt - pd.Timedelta(days=day)
        end = stats_dt
        has_positive = any((data[column] >= start) &
                            (data[column] <= end))
        return has_positive

    # 入群前3天猪只检出抗原阳性
    def pig_check_out_3(self, pigfarm_dk):
        # 获取对应猪场的数据
        check_out_data = self.intro_tame_data[self.intro_tame_data['prorg_inv_dk'] == pigfarm_dk]

        # 获取入群前3天猪只检出阳性的数据
        check_out_data = check_out_data[(check_out_data['rqbe3_blue_ear_kyyd_check_out_qty'] > 0) |
                                        (check_out_data['rqbe3_blue_ear_kypt_check_out_qty'] > 0)]

        # 如果没有检出阳性数据，直接返回
        if check_out_data.empty:
            return

        # 获取该猪场的所有样本日期
        farm_samples = self.index_sample[self.index_sample['pigfarm_dk'] == pigfarm_dk]

        # 对每个样本日期进行处理
        for _, sample in farm_samples.iterrows():
            stats_dt = sample['stats_dt']
            sample_idx = self.index_sample[(self.index_sample['pigfarm_dk'] == pigfarm_dk) &
                                      (self.index_sample['stats_dt'] == stats_dt)].index

            # 判断abort_8_14_decision: stats_dt - 21 到 stats_dt
            has_positive_8_14 = self.judge_contition(stats_dt, 21, check_out_data, 'min_boar_inpop_dt')
            # 判断abort_15_21_decision: stats_dt - 14到stats_dt
            has_positive_15_21 = self.judge_contition(stats_dt, 14, check_out_data, 'min_boar_inpop_dt')

            # 只有当检测为阳性时才将决策值设置为1
            if has_positive_8_14:
                self.index_sample.loc[sample_idx, 'abort_8_14_decision'] = 1
            if has_positive_15_21:
                self.index_sample.loc[sample_idx, 'abort_15_21_decision'] = 1


    # 超100头猪苗积压，且猪苗天龄大于40
    def piglet_overstock(self, pigfarm_dk):
        # 获取对应猪场的数据
        piglet_data = self.piglets_data[self.piglets_data['org_farm_dk'] == pigfarm_dk]

        # 筛选出天龄大于40且猪苗数量大于100的数据
        overstock_data = piglet_data[(piglet_data['pd03010103'] > 40) & (piglet_data['pd25010316'] > 100)]

        # 筛选日期范围
        overstock_data = overstock_data[(overstock_data['stats_dt'] >= self.running_start_date - pd.Timedelta(days=30)) &
                                        (overstock_data['stats_dt'] <= self.running_end_date)]

                # 如果没有检出阳性数据，直接返回
        if overstock_data.empty:
            return

        # 获取该猪场的所有样本日期
        farm_samples = self.index_sample[self.index_sample['pigfarm_dk'] == pigfarm_dk]

        # 对每个样本日期进行处理
        for _, sample in farm_samples.iterrows():
            stats_dt = sample['stats_dt']
            sample_idx = self.index_sample[(self.index_sample['pigfarm_dk'] == pigfarm_dk) &
                                      (self.index_sample['stats_dt'] == stats_dt)].index

            # 判断abort_8_14_decision: stats_dt - 21到stats_dt
            has_positive_8_14 = self.judge_contition(stats_dt, 21, overstock_data, 'stats_dt')
            # 判断abort_15_21_decision: stats_dt - 14到stats_dt
            has_positive_15_21 = self.judge_contition(stats_dt, 14, overstock_data, 'stats_dt')

            # 只有当检测为阳性时才将决策值设置为1
            if has_positive_8_14:
                self.index_sample.loc[sample_idx, 'abort_8_14_decision'] = 1
            if has_positive_15_21:
                self.index_sample.loc[sample_idx, 'abort_15_21_decision'] = 1


    # 引种前8-30天猪只蓝耳（野毒）检出阳性
    def pig_check_out_8_30(self, pigfarm_dk):
        # 获取对应猪场的数据
        check_out_data = self.intro_tame_data[self.intro_tame_data['prorg_inv_dk'] == pigfarm_dk]

        # 引种前8-30天猪只蓝耳（野毒）检出阳性
        check_out_data = check_out_data[check_out_data['yzbe8_blue_ear_kyyd_check_out_qty'] > 0]
        # 筛选日期范围
        check_out_data = check_out_data[(check_out_data['allot_dt'] >= self.running_start_date - pd.Timedelta(days=30)) &
                                          (check_out_data['allot_dt'] <= self.running_end_date)]

        # 如果没有检出阳性数据，直接返回
        if check_out_data.empty:
            return

        # 获取该猪场的所有样本日期
        farm_samples = self.index_sample[self.index_sample['pigfarm_dk'] == pigfarm_dk]

        # 对每个样本日期进行处理
        for _, sample in farm_samples.iterrows():
            stats_dt = sample['stats_dt']
            sample_idx = self.index_sample[(self.index_sample['pigfarm_dk'] == pigfarm_dk) &
                                      (self.index_sample['stats_dt'] == stats_dt)].index

            # 盘算abort_8_14_decision: stats_dt - 21到stats_dt
            has_positive_8_14 = self.judge_contition(stats_dt, 21, check_out_data, 'allot_dt')
            # 判断abort_15_21_decision: stats_dt - 14到stats_dt
            has_positive_15_21 = self.judge_contition(stats_dt, 14, check_out_data, 'allot_dt')

            # 更新决策值
            # 只有当检测为阳性时才将决策值设置为1
            if has_positive_8_14:
                self.index_sample.loc[sample_idx, 'abort_8_14_decision'] = 1
            if has_positive_15_21:
                self.index_sample.loc[sample_idx, 'abort_15_21_decision'] = 1


    def get_result(self):
        # 获取index_sample的所有猪场数据键
        pigfarm_dks = self.index_sample['pigfarm_dk'].unique()

        # 初始化所有决策值为0
        self.index_sample['abort_1_7_decision'] = 0
        self.index_sample['abort_8_14_decision'] = 0
        self.index_sample['abort_15_21_decision'] = 0

        for pigfarm_dk in tqdm(pigfarm_dks, desc="Processing pig farms"):
            # 处理每个猪场的数据
            self.pig_check_out_3(pigfarm_dk)
            self.piglet_overstock(pigfarm_dk)
            self.pig_check_out_8_30(pigfarm_dk)

        self.index_sample['abort_1_7_pred'] = self.index_sample['abort_1_7_decision']
        self.index_sample['abort_8_14_pred'] = self.index_sample['abort_8_14_decision']
        self.index_sample['abort_15_21_pred'] = self.index_sample['abort_15_21_decision']

        return self.index_sample


if __name__ == "__main__":
    for start_date, end_date, i in [
        ('2024-03-01', '2024-03-30', 3),
        ('2024-06-01', '2024-06-30', 6),
        ('2024-09-01', '2024-09-30', 9),
        ('2024-12-01', '2024-12-30', 12),
    ]:
        index_sample, index_ground_truth = abortion_abnormal_index_sample_v2(start_date, end_date)
        baseline = RuleBaseline(start_date, end_date, index_sample)
        result = baseline.get_result()
        save_path = f"data/predict/abort_abnormal/v1.0.49 rule_baseline_v2/v1.0.49 rule_baseline_v2 {i}"
        os.makedirs(save_path, exist_ok=True)
        result.to_csv(f"{save_path}/abort_abnormal.csv", index=False, encoding='utf-8')

