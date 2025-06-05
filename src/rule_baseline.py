import pandas as pd
from tqdm import tqdm

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

class Baseline:

    def __init__(self, running_start_date, running_end_date):
        self.running_start_date = pd.to_datetime(running_start_date)
        self.running_end_date = pd.to_datetime(running_end_date)
        self.get_data()


    def get_data(self):
        # 获取数据
        self.piglets_data = pd.read_csv("data/raw_data/ads_pig_efficient_piglet_batch_analysis_day.csv", encoding='utf-8')
        self.piglets_data['stats_dt'] = pd.to_datetime(self.piglets_data['stats_dt'], format='mixed')
        self.piglets_data = self.piglets_data[(self.piglets_data['stats_dt'] >= self.running_start_date - pd.Timedelta(days=30)) &
                                              (self.piglets_data['stats_dt'] <= self.running_end_date)]

        self.intro_tame_data = pd.read_csv("data/raw_data/ads_pig_isolation_tame_proline_risk.csv", encoding='utf-8', low_memory=False)
        self.intro_tame_data['allot_dt'] = pd.to_datetime(self.intro_tame_data['allot_dt'])
        self.intro_tame_data['min_boar_inpop_dt'] = pd.to_datetime(self.intro_tame_data['min_boar_inpop_dt'])

        self.index_sample = pd.read_csv("data/interim_data/eval/abortion_abnormal_eval_data/abortion_abnormal_index_sample.csv", encoding='utf-8')
        self.index_sample['stats_dt'] = pd.to_datetime(self.index_sample['stats_dt'])
        self.index_sample.sort_values(by=['pigfarm_dk', 'stats_dt'], inplace=True)


    # 入群前3天猪只检出抗原阳性
    def pig_check_out_3(self, pigfarm_dk):
        # 获取对应猪场的数据
        check_out_data = self.intro_tame_data[self.intro_tame_data['prorg_inv_dk'] == pigfarm_dk]
        
        # 获取入群前3天猪只检出阳性的数据
        check_out_data = check_out_data[(check_out_data['rqbe3_blue_ear_kyyd_check_out_qty'] > 0) | 
                                        (check_out_data['rqbe3_blue_ear_kypt_check_out_qty'] > 0)]
        
        # 筛选日期范围
        check_out_data = check_out_data[(check_out_data['min_boar_inpop_dt'] >= self.running_start_date - pd.Timedelta(days=30)) &
                                          (check_out_data['min_boar_inpop_dt'] <= self.running_end_date)]

        # 如果没有检出阳性数据，直接返回
        if check_out_data.empty:
            return
            
        # 获取所有入群日期
        entry_dates = check_out_data['min_boar_inpop_dt'].dropna().unique()
        
        # 获取该猪场的所有样本数据
        farm_samples = self.index_sample[self.index_sample['pigfarm_dk'] == pigfarm_dk]
        if farm_samples.empty:
            return
        
        # 对每个入群日期处理
        for entry_date in entry_dates:
            # 计算每个样本日期与入群日期的天数差
            for idx, sample in farm_samples.iterrows():
                days_diff = (sample['stats_dt'] - entry_date).days
                
                # 根据天数差设置相应的abort字段
                if 0 <= days_diff <= 6:
                    self.index_sample.loc[idx, ['abort_1_7_decision', 'abort_8_14_decision', 'abort_15_21_decision']] = 1
                elif 7 <= days_diff <= 13:
                    self.index_sample.loc[idx, ['abort_1_7_decision', 'abort_8_14_decision']] = 1
                elif 14 <= days_diff <= 20:
                    self.index_sample.loc[idx, 'abort_1_7_decision'] = 1


    # 超100头猪苗积压，且猪苗天龄大于40
    def piglet_overstock(self, pigfarm_dk):
        # 获取对应猪场的数据
        piglet_data = self.piglets_data[self.piglets_data['org_farm_dk'] == pigfarm_dk]

        # 筛选出天龄大于40且猪苗数量大于100的数据
        overstock_data = piglet_data[(piglet_data['pd03010103'] > 40) & (piglet_data['pd25010316'] > 100)]

        # 筛选日期范围
        overstock_data = overstock_data[(overstock_data['stats_dt'] >= self.running_start_date - pd.Timedelta(days=30)) &
                                        (overstock_data['stats_dt'] <= self.running_end_date)]

        # 如果没有积压数据，直接返回
        if overstock_data.empty:
            return
        
        # 获取所有积压日期
        overstock_dates = overstock_data['stats_dt'].dropna().unique()
        
        # 获取该猪场的所有样本数据
        farm_samples = self.index_sample[self.index_sample['pigfarm_dk'] == pigfarm_dk]
        if farm_samples.empty:
            return
        
        # 对每个积压日期处理
        for overstock_date in overstock_dates:
            # 计算每个样本日期与积压日期的天数差
            for idx, sample in farm_samples.iterrows():
                days_diff = (sample['stats_dt'] - overstock_date).days

                # 根据天数差设置相应的abort字段
                if 0 <= days_diff <= 6:
                    self.index_sample.loc[idx, ['abort_1_7_decision', 'abort_8_14_decision', 'abort_15_21_decision']] = 1
                elif 7 <= days_diff <= 13:
                    self.index_sample.loc[idx, ['abort_1_7_decision', 'abort_8_14_decision']] = 1
                elif 14 <= days_diff <= 20:
                    self.index_sample.loc[idx, 'abort_1_7_decision'] = 1


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
            
        # 获取所有引种日期
        entry_dates = check_out_data['allot_dt'].dropna().unique()
        
        # 获取该猪场的所有样本数据
        farm_samples = self.index_sample[self.index_sample['pigfarm_dk'] == pigfarm_dk]
        if farm_samples.empty:
            return
        
        # 对每个引种日期处理
        for entry_date in entry_dates:
            # 计算每个样本日期与引种日期的天数差
            for idx, sample in farm_samples.iterrows():
                days_diff = (sample['stats_dt'] - entry_date).days
                
                # 根据天数差设置相应的abort字段
                if 0 <= days_diff <= 6:
                    self.index_sample.loc[idx, ['abort_1_7_decision', 'abort_8_14_decision', 'abort_15_21_decision']] = 1
                elif 7 <= days_diff <= 13:
                    self.index_sample.loc[idx, ['abort_1_7_decision', 'abort_8_14_decision']] = 1
                elif 14 <= days_diff <= 20:
                    self.index_sample.loc[idx, 'abort_1_7_decision'] = 1


    def get_result(self):
        # 初始化结果
        self.index_sample['abort_1_7_decision'] = 0
        self.index_sample['abort_8_14_decision'] = 0
        self.index_sample['abort_15_21_decision'] = 0

        # 获取index_sample的所有猪场数据键
        pigfarm_dks = self.index_sample['pigfarm_dk'].unique()

        for pigfarm_dk in tqdm(pigfarm_dks):
            # 处理每个猪场的数据
            self.pig_check_out_3(pigfarm_dk)
            self.piglet_overstock(pigfarm_dk)
            self.pig_check_out_8_30(pigfarm_dk)
        
        self.index_sample['abort_1_7_pred'] = self.index_sample['abort_1_7_decision']
        self.index_sample['abort_8_14_pred'] = self.index_sample['abort_8_14_decision']
        self.index_sample['abort_15_21_pred'] = self.index_sample['abort_15_21_decision']

        return self.index_sample


if __name__ == "__main__":
    baseline = Baseline("2024-03-01", "2024-03-30")
    index_sample = baseline.get_result()
    index_sample.to_csv("data/predict/rule_baseline/abort_abnormal.csv", index=False, encoding='utf-8-sig')

