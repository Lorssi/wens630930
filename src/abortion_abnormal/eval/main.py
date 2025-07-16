import pandas as pd
import logging
import os
import sys
from pathlib import Path

# 获取项目根目录（假设是 src 的父目录）
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_root))

from abortion_abnormal.eval.eval_base import EvalBaseMixin
from abortion_abnormal.eval.build_eval_dateset import abortion_abnormal_index_sample, abortion_days_index_sample, abortion_abnormal_index_sample_v2
from abortion_abnormal.eval.abortion_abnormal_predict_eval_data import AbortionAbnormalPredictEvalData, AbortionAbnormalPredictEvalData_v3
from abortion_abnormal.eval.abortion_days_predict_eval_data import AbortionDaysPredictEvalData
import abortion_abnormal.eval.eval_config as config
from split_data_to_one_pigfarm import split_data_to_one_pigfarm

# base_dir = os.path.split(os.path.realpath(__file__))[0]
program = os.path.basename(sys.argv[0])
logger = logging.getLogger(program)
logging.basicConfig(format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s: %(message)s',
                    datefmt='%a, %d %b %Y %H:%M:%S')
logging.root.setLevel(level=logging.INFO)
# warnings.filterwarnings("ignore")

# 任务1评估-流产率异常预测评估
class AbortionAbnormalPredictEval(EvalBaseMixin):
    def __init__(self, logger=None):
        self.logger = logger
        self.pigfarm_dk = None
        self.pigfarm_dks = None
        self.eval_running_dt_start = None
        self.eval_running_dt_end = None
        self.index_sample = None
        self.sample_ground_truth = None
        self.predict_data = None


    # 构建评测数据集
    # def build_eval_set(self, eval_running_dt, eval_interval, param=None, predict_data=None, pigfarm_dk=None, use_cache = False):
    def build_eval_set(self, eval_running_dt, eval_interval, param=None, predict_data=None, pigfarm_dk=None, use_cache = False, pigfarm_dks=None):
        self.pigfarm_dk = pigfarm_dk

        eval_running_dt = pd.to_datetime(eval_running_dt)
        self.eval_running_dt_start = (eval_running_dt + pd.Timedelta(days=1)).strftime('%Y-%m-%d')
        self.eval_running_dt_end = (eval_running_dt + pd.Timedelta(days=eval_interval)).strftime('%Y-%m-%d')

        # 获取真实值
        self.index_sample, self.sample_ground_truth = abortion_abnormal_index_sample(self.eval_running_dt_start, self.eval_running_dt_end, use_cache=use_cache)
        self.index_sample.to_csv(config.abortion_abnormal_eval_index_sample_save_path, index=False, encoding='utf-8')
        self.sample_ground_truth.to_csv(config.abortion_abnormal_eval_ground_truth_save_path, index=False, encoding='utf-8')


        if pigfarm_dks is not None:
            self.pigfarm_dks = pigfarm_dks
            pdks = [dk.replace('@', '/') for dk in pigfarm_dks] # 替换回原来的斜杠格式
            # 如果传入了猪场列表，则只保留这些猪场的数据
            self.index_sample = self.index_sample[self.index_sample['pigfarm_dk'].isin(pdks)].copy()
            self.sample_ground_truth = self.sample_ground_truth[self.sample_ground_truth['pigfarm_dk'].isin(pdks)].copy()

        # delete 获取预测结果
        feature = config.EvalFilename.feature
        if feature is None:
            predict_file_path = "data/predict/abort_abnormal/abort_abnormal.csv"
        else:
            predict_file_path = f"data/predict/abort_abnormal/{config.EvalFilename.version} {feature}/{config.EvalFilename.version} {feature} {config.EvalFilename.eval_date_month}/abort_abnormal.csv"
        if predict_data is not None:
            # 如果传入了预测数据，则使用传入的数据
            self.predict_data = predict_data
        else:
            self.predict_data = pd.read_csv(predict_file_path, encoding='utf-8')

        # delete 转化日期格式
        self.predict_data['stats_dt'] = pd.to_datetime(self.predict_data['stats_dt'])

        # delete  ############################################################################################################
        # self.sample_ground_truth = self.predict_data[['pigfarm_dk', 'stats_dt', 'abort_1_7', 'abort_8_14', 'abort_15_21']]
        # self.predict_data.drop(columns=['abort_1_7', 'abort_8_14', 'abort_15_21'], inplace=True, errors='ignore')
        # delete  ############################################################################################################

        # 获取组织分类表
        dim_org_inv = pd.read_csv("data/raw_data/dim_org_inv.csv", encoding='utf-8')

        # 处理组织分类信息，只保留需要的列
        org_mapping = dim_org_inv[['org_inv_dk', 'l2_org_inv_nm', 'l3_org_inv_nm', 'l4_org_inv_nm']].copy()

        # 确保ID列为字符串类型以避免join时的类型不匹配问题
        org_mapping['org_inv_dk'] = org_mapping['org_inv_dk'].astype(str)

        # 将与组织数据合并，添加部门信息
        self.sample_ground_truth = self.sample_ground_truth.merge(
            org_mapping,
            left_on='pigfarm_dk',
            right_on='org_inv_dk',
            how='left'
        )

        # 筛选数据
        if pigfarm_dk is not None:
            pdk = pigfarm_dk.replace('@', '/') # 替换回原来的斜杠格式
            # 如果指定了特定的猪场，则只保留这些猪场的数据
            self.predict_data = self.predict_data[self.predict_data['pigfarm_dk'] == pdk].copy()
            self.sample_ground_truth = self.sample_ground_truth[self.sample_ground_truth['pigfarm_dk'] == pdk].copy()


        # tmp = self.sample_ground_truth.merge(
        #     self.predict_data,
        #     on=['pigfarm_dk', 'stats_dt'],
        #     how='left'
        # )
        # tmp.to_csv(f'tmp_{feature}.csv', index=False, encoding='utf-8')



    def get_eval_index_sample(self):
        return self.index_sample


    def eval_with_index_sample(self, save_flag=True):
        eval_dt = AbortionAbnormalPredictEvalData(self.predict_data, self.sample_ground_truth, self.eval_running_dt_start, self.eval_running_dt_end, logger)

        # 整体评估
        all_sample_result = eval_dt.eval_all_samples()
        # 整体评估-剔除非瘟数据
        all_sample_result_exclude_feiwen = eval_dt.eval_all_samples_exclude_feiwen()
        # 特殊样本分析
        special_samples_result = eval_dt.eval_special_samples()
        if self.pigfarm_dk is None and self.pigfarm_dks is None:
            # 特殊样本分析-猪业一部
            special_samples_result2 = eval_dt.eval_special_samples(l2_name=['猪业一部'])
            # 分级组织评估
            l2_results, l3_results, l4_results = eval_dt.eval_organizational_hierarchy()
            # 流产持续时长统计
            abortion_duration_results = eval_dt.calculate_abortion_duration()
            # 流产间隔统计
            abortion_interval_results = eval_dt.calculate_abortion_interval()

        # delete 临时组合，复制方便
        data = all_sample_result_exclude_feiwen.copy()
        # 定义期望的eval_period顺序
        expected_periods = ['1_7', '8_14', '15_21']

        # 将specail_samples_result[sample_type]=='normal-2_to_abnormal-2'的precision,recall,f1_score,auc,special_recall列加在data后面
        if 'sample_type' in special_samples_result.columns:
            normal_to_abnormal = special_samples_result[special_samples_result['sample_type'] == 'normal-2_to_abnormal-2'].reset_index(drop=True).copy()
            # 处理normal_to_abnormal数据，确保按期望顺序
            for col in ['precision', 'recall', 'f1_score', 'auc', 'special_recall']:
                values = []
                for period in expected_periods:
                    if not normal_to_abnormal.empty and 'eval_period' in normal_to_abnormal.columns:
                        period_data = normal_to_abnormal[normal_to_abnormal['eval_period'] == period]
                        if not period_data.empty and col in period_data.columns:
                            values.append(period_data[col].iloc[0])
                        else:
                            values.append(None)
                    else:
                        values.append(None)
                data[f'normal_to_abnormal_{col}'] = values

            # 将specail_samples_result[sample_type]=='abnormal-2_to_normal-2'的precision,recall,f1_score,auc,special_recall列加在data后面
            abnormal_to_normal = special_samples_result[special_samples_result['sample_type'] == 'abnormal-2_to_normal-2'].reset_index().copy()
            # 处理abnormal_to_normal数据，确保按期望顺序
            for col in ['precision', 'recall', 'f1_score', 'auc', 'special_recall']:
                values = []
                for period in expected_periods:
                    if not abnormal_to_normal.empty and 'eval_period' in abnormal_to_normal.columns:
                        period_data = abnormal_to_normal[abnormal_to_normal['eval_period'] == period]
                        if not period_data.empty and col in period_data.columns:
                            values.append(period_data[col].iloc[0])
                        else:
                            values.append(None)
                    else:
                        values.append(None)
                data[f'abnormal_to_normal_{col}'] = values
            # 将specail_samples_result[sample_type]=='abnormal_to_abnormal'的precision,recall,f1_score,auc,special_recall列加在data后面
            abnormal_to_abnormal = special_samples_result[special_samples_result['sample_type'] == 'abnormal_to_abnormal'].reset_index().copy()
            # 处理abnormal_to_abnormal数据，确保按期望顺序
            for col in ['precision', 'recall', 'f1_score', 'auc', 'special_recall']:
                values = []
                for period in expected_periods:
                    if not abnormal_to_abnormal.empty and 'eval_period' in abnormal_to_abnormal.columns:
                        period_data = abnormal_to_abnormal[abnormal_to_abnormal['eval_period'] == period]
                        if not period_data.empty and col in period_data.columns:
                            values.append(period_data[col].iloc[0])
                        else:
                            values.append(None)
                    else:
                        values.append(None)
                data[f'abnormal_to_abnormal_{col}'] = values
        else:
            for col in ['precision', 'recall', 'f1_score', 'auc', 'special_recall']:
                data[f'normal_to_abnormal_{col}'] = None
                data[f'abnormal_to_normal_{col}'] = None
                data[f'abnormal_to_abnormal_{col}'] = None
        if self.pigfarm_dk is None and self.pigfarm_dks is None:
            # 将l2_results[l2_name]=='猪业一部'的stats_dt,l2_name,total_sample_num,remain_sample_num,eval_period,precision,recall,f1_score,auc,recognition列加在data后面
            pig_dept = l2_results[l2_results['l2_name'] == '猪业一部'].reset_index().copy()
            if not pig_dept.empty:
                cols_to_add = ['stats_dt', 'l2_name', 'total_sample_num', 'remain_sample_num',
                            'eval_period', 'precision', 'recall', 'f1_score', 'auc', 'recognition']
                for col in cols_to_add:
                    if col in pig_dept.columns:
                        data[f'pig_dept_{col}'] = pig_dept[col]
            # 将specail_samples_result2[sample_type]=='normal-2_to_abnormal-2'的precision,recall,f1_score,auc,special_recall列加在data后面
            if 'sample_type' in special_samples_result2.columns:
                normal_to_abnormal2 = special_samples_result2[special_samples_result2['sample_type'] == 'normal-2_to_abnormal-2'].reset_index().copy()
                if not normal_to_abnormal2.empty:
                    for col in ['precision', 'recall', 'f1_score', 'auc', 'special_recall']:
                        if col in normal_to_abnormal2.columns:
                            data[f'normal_to_abnormal_2_{col}'] = normal_to_abnormal2[col]


        # 保存结果
        if config.EvalFilename.feature is not None:
            suffix = f"{self.eval_running_dt_start}_{config.EvalFilename.feature}"
        else:
            suffix = self.eval_running_dt_start
        if save_flag:
            with pd.ExcelWriter(config.abortion_abnormal_eval_result_save_path.format(config.EvalFilename.version, suffix)) as writer:
                all_sample_result.to_excel(writer, sheet_name='整体', index=False)
                all_sample_result_exclude_feiwen.to_excel(writer, sheet_name='整体-剔除非瘟数据', index=False, float_format="%.4f")
                special_samples_result.to_excel(writer, sheet_name='特殊样本分析', index=False, float_format="%.4f")
                if self.pigfarm_dk is None and self.pigfarm_dks is None:
                    special_samples_result2.to_excel(writer, sheet_name='特殊样本分析-猪业一部', index=False, float_format="%.4f")
                    l2_results.to_excel(writer, sheet_name='二级组织分类', index=False, float_format="%.4f")
                    l3_results.to_excel(writer, sheet_name='三级组织分类', index=False, float_format="%.4f")
                    l4_results.to_excel(writer, sheet_name='四级组织分类', index=False, float_format="%.4f")
                    abortion_duration_results.to_excel(writer, sheet_name='流产持续时长统计', index=False, float_format="%.4f")
                    abortion_interval_results.to_excel(writer, sheet_name='流产间隔统计', index=False, float_format="%.4f")

        return data

# 任务1评估-流产率异常预测评估_v3
class AbortionAbnormalPredictEval_v3(EvalBaseMixin):
    def __init__(self, logger=None):
        self.logger = logger
        self.eval_start_dt = None
        self.eval_end_dt = None
        self.index_sample = None
        self.sample_ground_truth = None
        self.predict_data = None


    # 构建评测数据集
    def build_eval_set(self, eval_start_dt, eval_end_dt, param=None, predict_data=None, use_cache = False):

        self.eval_start_dt = eval_start_dt
        self.eval_end_dt = eval_end_dt

        # 获取真实值
        self.index_sample, self.sample_ground_truth = abortion_abnormal_index_sample(self.eval_start_dt, self.eval_end_dt, use_cache=use_cache)
        self.index_sample.to_csv(config.abortion_abnormal_eval_index_sample_save_path, index=False, encoding='utf-8')
        self.sample_ground_truth.to_csv(config.abortion_abnormal_eval_ground_truth_save_path, index=False, encoding='utf-8')

        # delete 获取预测结果
        feature = config.EvalFilename.feature
        if feature is None:
            predict_file_path = "data/predict/abort_abnormal/abort_abnormal.csv"
        else:
            predict_file_path = f"data/predict/abort_abnormal/{config.EvalFilename.version} {feature}/{config.EvalFilename.version} {feature} {config.EvalFilename.eval_date_month}/abort_abnormal.csv"
        if predict_data is not None:
            # 如果传入了预测数据，则使用传入的数据
            self.predict_data = predict_data
        else:
            self.predict_data = pd.read_csv(predict_file_path, encoding='utf-8')

        # delete 转化日期格式
        self.predict_data['stats_dt'] = pd.to_datetime(self.predict_data['stats_dt'])

        # delete  ############################################################################################################
        # self.sample_ground_truth = self.predict_data[['pigfarm_dk', 'stats_dt', 'abort_1_7', 'abort_8_14', 'abort_15_21']]
        # self.predict_data.drop(columns=['abort_1_7', 'abort_8_14', 'abort_15_21'], inplace=True, errors='ignore')
        # delete  ############################################################################################################

        # 获取组织分类表
        dim_org_inv = pd.read_csv("data/raw_data/dim_org_inv.csv", encoding='utf-8')

        # 处理组织分类信息，只保留需要的列
        org_mapping = dim_org_inv[['org_inv_dk', 'l2_org_inv_nm', 'l3_org_inv_nm', 'l4_org_inv_nm']].copy()

        # 确保ID列为字符串类型以避免join时的类型不匹配问题
        org_mapping['org_inv_dk'] = org_mapping['org_inv_dk'].astype(str)

        # 将与组织数据合并，添加部门信息
        self.sample_ground_truth = self.sample_ground_truth.merge(
            org_mapping,
            left_on='pigfarm_dk',
            right_on='org_inv_dk',
            how='left'
        )

        tmp = self.sample_ground_truth.merge(
            self.predict_data,
            on=['pigfarm_dk', 'stats_dt'],
            how='left'
        )
        tmp.to_csv(f'tmp_{feature}.csv', index=False, encoding='utf-8')


    def get_eval_index_sample(self):
        return self.index_sample


    def eval_with_index_sample(self, save_flag=True):
        eval_dt = AbortionAbnormalPredictEvalData_v3(self.predict_data, self.sample_ground_truth, self.eval_start_dt, self.eval_end_dt, logger)

        # 整体评估
        all_sample_result = eval_dt.eval_all_samples()
        # 整体评估-剔除非瘟数据
        all_sample_result_exclude_feiwen = eval_dt.eval_all_samples_exclude_feiwen()
        # 特殊样本分析
        special_samples_result = eval_dt.eval_special_samples()
        # 特殊样本分析-猪业一部
        special_samples_result2 = eval_dt.eval_special_samples(l2_name=['猪业一部'])
        # 分级组织评估
        l2_results, l3_results, l4_results = eval_dt.eval_organizational_hierarchy()
        # 流产持续时长统计
        abortion_duration_results = eval_dt.calculate_abortion_duration()
        # 流产间隔统计
        abortion_interval_results = eval_dt.calculate_abortion_interval()

        # delete 临时组合，复制方便
        data = all_sample_result_exclude_feiwen.copy()
        # 定义期望的eval_period顺序
        expected_periods = ['1_7', '8_14', '15_21']

        # 将specail_samples_result[sample_type]=='normal-2_to_abnormal-2'的precision,recall,f1_score,auc,special_recall列加在data后面
        if 'sample_type' in special_samples_result.columns:
            normal_to_abnormal = special_samples_result[special_samples_result['sample_type'] == 'normal-2_to_abnormal-2'].reset_index(drop=True).copy()
            # 处理normal_to_abnormal数据，确保按期望顺序
            for col in ['precision', 'recall', 'f1_score', 'auc', 'special_recall']:
                values = []
                for period in expected_periods:
                    if not normal_to_abnormal.empty and 'eval_period' in normal_to_abnormal.columns:
                        period_data = normal_to_abnormal[normal_to_abnormal['eval_period'] == period]
                        if not period_data.empty and col in period_data.columns:
                            values.append(period_data[col].iloc[0])
                        else:
                            values.append(None)
                    else:
                        values.append(None)
                data[f'normal_to_abnormal_{col}'] = values

            # 将specail_samples_result[sample_type]=='abnormal-2_to_normal-2'的precision,recall,f1_score,auc,special_recall列加在data后面
            abnormal_to_normal = special_samples_result[special_samples_result['sample_type'] == 'abnormal-2_to_normal-2'].reset_index().copy()
            # 处理abnormal_to_normal数据，确保按期望顺序
            for col in ['precision', 'recall', 'f1_score', 'auc', 'special_recall']:
                values = []
                for period in expected_periods:
                    if not abnormal_to_normal.empty and 'eval_period' in abnormal_to_normal.columns:
                        period_data = abnormal_to_normal[abnormal_to_normal['eval_period'] == period]
                        if not period_data.empty and col in period_data.columns:
                            values.append(period_data[col].iloc[0])
                        else:
                            values.append(None)
                    else:
                        values.append(None)
                data[f'abnormal_to_normal_{col}'] = values
            # 将specail_samples_result[sample_type]=='abnormal_to_abnormal'的precision,recall,f1_score,auc,special_recall列加在data后面
            abnormal_to_abnormal = special_samples_result[special_samples_result['sample_type'] == 'abnormal_to_abnormal'].reset_index().copy()
            # 处理abnormal_to_abnormal数据，确保按期望顺序
            for col in ['precision', 'recall', 'f1_score', 'auc', 'special_recall']:
                values = []
                for period in expected_periods:
                    if not abnormal_to_abnormal.empty and 'eval_period' in abnormal_to_abnormal.columns:
                        period_data = abnormal_to_abnormal[abnormal_to_abnormal['eval_period'] == period]
                        if not period_data.empty and col in period_data.columns:
                            values.append(period_data[col].iloc[0])
                        else:
                            values.append(None)
                    else:
                        values.append(None)
                data[f'abnormal_to_abnormal_{col}'] = values
        else:
            for col in ['precision', 'recall', 'f1_score', 'auc', 'special_recall']:
                data[f'normal_to_abnormal_{col}'] = None
                data[f'abnormal_to_normal_{col}'] = None
                data[f'abnormal_to_abnormal_{col}'] = None
        # 将l2_results[l2_name]=='猪业一部'的stats_dt,l2_name,total_sample_num,remain_sample_num,eval_period,precision,recall,f1_score,auc,recognition列加在data后面
        pig_dept = l2_results[l2_results['l2_name'] == '猪业一部'].reset_index().copy()
        if not pig_dept.empty:
            cols_to_add = ['stats_dt', 'l2_name', 'total_sample_num', 'remain_sample_num',
                        'eval_period', 'precision', 'recall', 'f1_score', 'auc', 'recognition']
            for col in cols_to_add:
                if col in pig_dept.columns:
                    data[f'pig_dept_{col}'] = pig_dept[col]
        # 将specail_samples_result2[sample_type]=='normal-2_to_abnormal-2'的precision,recall,f1_score,auc,special_recall列加在data后面
        if 'sample_type' in special_samples_result2.columns:
            normal_to_abnormal2 = special_samples_result2[special_samples_result2['sample_type'] == 'normal-2_to_abnormal-2'].reset_index().copy()
            if not normal_to_abnormal2.empty:
                for col in ['precision', 'recall', 'f1_score', 'auc', 'special_recall']:
                    if col in normal_to_abnormal2.columns:
                        data[f'normal_to_abnormal_2_{col}'] = normal_to_abnormal2[col]


        # 保存结果
        if config.EvalFilename.feature is not None:
            suffix = f"{self.eval_start_dt}_{config.EvalFilename.feature}"
        else:
            suffix = self.eval_start_dt
        if save_flag:
            with pd.ExcelWriter(config.abortion_abnormal_eval_result_save_path.format(config.EvalFilename.version, suffix)) as writer:
                all_sample_result.to_excel(writer, sheet_name='整体', index=False)
                all_sample_result_exclude_feiwen.to_excel(writer, sheet_name='整体-剔除非瘟数据', index=False, float_format="%.4f")
                special_samples_result.to_excel(writer, sheet_name='特殊样本分析', index=False, float_format="%.4f")
                special_samples_result2.to_excel(writer, sheet_name='特殊样本分析-猪业一部', index=False, float_format="%.4f")
                l2_results.to_excel(writer, sheet_name='二级组织分类', index=False, float_format="%.4f")
                l3_results.to_excel(writer, sheet_name='三级组织分类', index=False, float_format="%.4f")
                l4_results.to_excel(writer, sheet_name='四级组织分类', index=False, float_format="%.4f")
                abortion_duration_results.to_excel(writer, sheet_name='流产持续时长统计', index=False, float_format="%.4f")
                abortion_interval_results.to_excel(writer, sheet_name='流产间隔统计', index=False, float_format="%.4f")

        return data


# 任务2评估-流产天数预测评估
class AbortionDayPredictEval(EvalBaseMixin):
    def __init__(self, logger=None):
        self.logger = logger

    # 构建评测数据集
    def build_eval_set(self, eval_running_dt, eval_interval, param=None):
        eval_running_dt = pd.to_datetime(eval_running_dt)
        self.eval_running_dt_start = (eval_running_dt + pd.Timedelta(days=1)).strftime('%Y-%m-%d')
        self.eval_running_dt_end = (eval_running_dt + pd.Timedelta(days=eval_interval)).strftime('%Y-%m-%d')

        # 获取真实值
        self.index_sample, self.index_ground_truth = abortion_days_index_sample(self.eval_running_dt_start, eval_running_dt_end=self.eval_running_dt_end)
        self.index_sample.to_csv(config.abortion_days_eval_index_sample_save_path, index=False, encoding='utf-8')
        self.index_ground_truth.to_csv(config.abortion_days_eval_ground_truth_save_path, index=False, encoding='utf-8')

        # delete 获取预测结果
        feature = config.EvalFilename.feature
        if feature is None:
            predict_file_path = "data/predict/abort_days/abort_abnormal_day.csv"
        else:
            predict_file_path = f"data/predict/abort_days/{config.EvalFilename.version} {feature}/{config.EvalFilename.version} {feature} {config.EvalFilename.eval_date_month}/abort_abnormal_day.csv"
        self.predict_data = pd.read_csv(predict_file_path, encoding='utf-8')
        # delete 转化日期格式
        self.predict_data['stats_dt'] = pd.to_datetime(self.predict_data['stats_dt'])
        # delete 转换列名
        self.predict_data.rename(columns={'abort_days_1_7': 'abort_days_1_7_decision',
                                          'abort_days_8_14': 'abort_days_8_14_decision',
                                          'abort_days_15_21': 'abort_days_15_21_decision',}, inplace=True)
        if self.predict_data.empty:
            self.logger.warning("预测数据为空，请检查预测结果文件路径或内容")
            return pd.DataFrame(), pd.DataFrame()

        # 获取组织分类表
        dim_org_inv = pd.read_csv("data/raw_data/dim_org_inv.csv", encoding='utf-8')

        # 处理组织分类信息，只保留需要的列
        org_mapping = dim_org_inv[['org_inv_dk', 'l2_org_inv_nm', 'l3_org_inv_nm', 'l4_org_inv_nm']].copy()

        # 确保ID列为字符串类型以避免join时的类型不匹配问题
        org_mapping['org_inv_dk'] = org_mapping['org_inv_dk'].astype(str)
        self.predict_data['pigfarm_dk'] = self.predict_data['pigfarm_dk'].astype(str)

        # 将与组织数据合并，添加部门信息
        self.index_ground_truth = self.index_ground_truth.merge(
            org_mapping,
            left_on='pigfarm_dk',
            right_on='org_inv_dk',
            how='left'
        )


    def get_eval_index_sample(self):
        return self.index_sample


    def eval_with_index_sample(self, predict_data=None, save_flag=True):
        if predict_data is None:
            predict_data = self.predict_data
        eval_dt = AbortionDaysPredictEvalData(predict_data, self.index_ground_truth, self.eval_running_dt_start, self.eval_running_dt_end, self.logger)

        # 整体评估
        all_sample_result = eval_dt.eval_all_samples()
        # 整体评估-剔除非瘟数据
        all_sample_result_exclude_feiwen, conf_matrix = eval_dt.eval_all_samples_exclude_feiwen()
        # 特殊样本分析
        special_samples_result = eval_dt.eval_special_samples()
        # 分级组织评估
        l2_results, l3_results, l4_results = eval_dt.eval_organizational_hierarchy()
        # 流产持续时长统计
        abortion_duration_results = eval_dt.calculate_abortion_duration()
        # 流产间隔统计
        abortion_interval_results = eval_dt.calculate_abortion_interval()
        # label数量统计
        label_count = eval_dt.calculate_label_num()
        # 混淆矩阵
        labels = list(range(8))
        combined_conf_matrix = pd.DataFrame()
        for period, matrix in conf_matrix.items():
            conf_df = pd.DataFrame(matrix, columns=labels)
            # 添加标签
            conf_df.insert(0, 'ground_truth', [i for i in labels])
            # 添加周期标识列
            conf_df.insert(0, 'period', period)
            # 垂直合并
            combined_conf_matrix = pd.concat([combined_conf_matrix, conf_df], ignore_index=True)
            # 添加空行分隔
            if period != list(conf_matrix.keys())[-1]:  # 不是最后一个
                empty_row = pd.DataFrame([[''] * len(conf_df.columns)], columns=conf_df.columns)
                combined_conf_matrix = pd.concat([combined_conf_matrix, empty_row], ignore_index=True)

        # 保存结果
        if config.EvalFilename.feature is not None:
            suffix = f"{self.eval_running_dt_start}_{config.EvalFilename.feature}"
        else:
            suffix = self.eval_running_dt_start
        if save_flag:
            with pd.ExcelWriter(config.abortion_days_eval_result_save_path.format(config.EvalFilename.version, suffix)) as writer:
                all_sample_result.to_excel(writer, sheet_name='整体', index=False, float_format="%.4f")
                all_sample_result_exclude_feiwen.to_excel(writer, sheet_name='整体-剔除非瘟数据', index=False, float_format="%.4f")
                special_samples_result.to_excel(writer, sheet_name='特殊样本分析', index=False, float_format="%.4f")
                l2_results.to_excel(writer, sheet_name='二级组织分类', index=False, float_format="%.4f")
                l3_results.to_excel(writer, sheet_name='三级组织分类', index=False, float_format="%.4f")
                l4_results.to_excel(writer, sheet_name='四级组织分类', index=False, float_format="%.4f")
                abortion_duration_results.to_excel(writer, sheet_name='流产持续时长统计', index=False, float_format="%.4f")
                abortion_interval_results.to_excel(writer, sheet_name='流产间隔统计', index=False, float_format="%.4f")
                label_count.to_excel(writer, sheet_name='标签数量统计', index=False, float_format="%.4f")
                combined_conf_matrix.to_excel(writer, sheet_name='混淆矩阵', index=False, float_format="%.4f")


# 任务3评估-流产率预测评估
class AbortionAbnormalEval(EvalBaseMixin):
    pass


def all_pigfarm_evaluate(version=None, features=None, predict_data=None, pigfarm_dks=None):
    if version is None:
        version = [
            'v1.0.66',
            'v1.0.67',
            'v1.0.68',
            'v1.0.69',
        ]

    if features is None:
        features = [
            'immu_count_30d',
            'immu_avg_delay_days_30d',
            'immu_delay_count_30d',
            'immu_delay_ratio_30d'
        ]

    for feature, version in zip(features, version):
        eval_result = pd.DataFrame()
        config.EvalFilename.feature = f'{feature}'
        config.EvalFilename.version = version
        for (start_date, month) in [
            ('2024-02-29', '3'),
            ('2024-05-31', '6'),
            ('2024-08-31', '9'),
            ('2024-11-30', '12'),
        ]:
            config.EvalFilename.eval_date_month = month
            abortion_abnormal_eval = AbortionAbnormalPredictEval(logger)
            abortion_abnormal_eval.build_eval_set(start_date, 30, predict_data=predict_data)
            # abortion_abnormal_eval.build_eval_set(start_date, 30, predict_data=predict_data, use_cache=True, pigfarm_dks=pigfarm_dks)
            data = abortion_abnormal_eval.eval_with_index_sample()
            eval_result = pd.concat([eval_result, data], ignore_index=True)

        eval_result.to_csv(f'{config.EvalFilename.version}_nfm_{config.EvalFilename.feature}.csv', index=False, encoding='utf-8', float_format="%.4f")

def all_pigfarm_evaluate_v3(version=None, features=None, predict_data=None, pigfarm_dks=None):
    if version is None:
        version = [
            'v1.0.0',
        ]

    if features is None:
        features = [
            'rule_baseline',
        ]

    for feature, version in zip(features, version):
        eval_result = pd.DataFrame()
        config.EvalFilename.feature = f'{feature}'
        config.EvalFilename.version = version
        for (start_date, end_date, month) in [
            ('2024-03-01', '2024-04-30', '3'),
            ('2024-06-01', '2024-07-31', '6'),
            ('2024-09-01', '2024-10-31', '9'),
            ('2024-12-01', '2025-01-31', '12'),
        ]:
            config.EvalFilename.eval_date_month = month
            abortion_abnormal_eval = AbortionAbnormalPredictEval_v3(logger)
            abortion_abnormal_eval.build_eval_set(start_date, end_date, predict_data=predict_data)
            data = abortion_abnormal_eval.eval_with_index_sample()
            eval_result = pd.concat([eval_result, data], ignore_index=True)

        eval_result.to_csv(f'{config.EvalFilename.version}_nfm_{config.EvalFilename.feature}.csv', index=False, encoding='utf-8', float_format="%.4f")


def per_pigfarm_evaluate():
    versions = [
        'v1.0.r',
        'v1.0.b',
        'v1.0.t'
    ]

    print(len(pd.read_csv('data/interim_data/eval/abortion_abnormal_eval_data/abortion_abnormal_ground_truth.csv')['pigfarm_dk'].unique()))

    index = pd.read_csv('pigfarm_dks.csv')
    pigfarm_dks = index['pigfarm_dk'].unique().tolist()
    # pigfarm_dks = ['L8X1mwESEADgADTRfwAAAcznrtQ=', 'bDoAAAg6uQPM567U', 'bDoAADt50rLM567U', 'bDoAAF5Fd63M567U']
    print(len(pigfarm_dks))
    # pigfarm_dks = ['bDoAAHYh1WTM567U', 'bDoAAAv2QUnM567U']
    pigfarm_dks = [dk.replace('/', '@') for dk in pigfarm_dks]

    # 测评策略基线
    rule_baseline_predict_data = pd.read_csv('data/predict/abort_abnormal/v1.0.0 rule_baseline/v1.0.0 rule_baseline 12/abort_abnormal.csv')
    all_pigfarm_evaluate(version=['v1.0.r'], features=['rule_baseline'], predict_data=rule_baseline_predict_data, pigfarm_dks=pigfarm_dks)
    # 测评最佳基线
    best_baseline_predict_data = pd.read_csv('data/predict/abort_abnormal/v1.0.0 best_baseline/v1.0.0 best_baseline 12/abort_abnormal.csv')
    all_pigfarm_evaluate(version=['v1.0.b'], features=['best_baseline'], predict_data=best_baseline_predict_data, pigfarm_dks=pigfarm_dks)
    # # 测评训练后数据
    train_predict_data = pd.read_csv('data/predict/abort_abnormal/v1.0.t train/v1.0.t train 12/abort_abnormal.csv')
    all_pigfarm_evaluate(version=['v1.0.t'], features=['train'], predict_data=train_predict_data, pigfarm_dks=pigfarm_dks)

    overall_col = ['stats_dt', 'total_sample_num', 'remain_sample_num', 'eval_period', 'precision', 'recall', 'f1_score', 'auc', 'recognition']
    prefixes = ['normal_to_abnormal', 'abnormal_to_normal', 'abnormal_to_abnormal']
    metrics = ['precision', 'recall', 'f1_score', 'auc', 'special_recall']
    special_col = [f"{prefix}_{metric}" for prefix in prefixes for metric in metrics ]

    all_result = {}

    for version in versions:
        for (start_date, month) in [
                # ('2024-02-29', '3'),
                # ('2024-05-31', '6'),
                # ('2024-08-31', '9'),
                ('2024-11-30', '12'),
            ]:
            config.EvalFilename.eval_date_month = month
            # 拆分策略基线数据评测
            if version == 'v1.0.r':
                predict_data_path = f"data/predict/abort_abnormal/v1.0.0 rule_baseline/v1.0.0 rule_baseline {month}/abort_abnormal.csv"
                splitter = split_data_to_one_pigfarm(pigfarm_dks, version, predict_data_path)
                splitter.split_data()
            elif version == 'v1.0.b':
                predict_data_path = f"data/predict/abort_abnormal/v1.0.0 best_baseline/v1.0.0 best_baseline {month}/abort_abnormal.csv"
                splitter = split_data_to_one_pigfarm(pigfarm_dks, version, predict_data_path)
                splitter.split_data()
            elif version == 'v1.0.t':
                # 暂时留着
                predict_data_path = None

            eval_result = pd.DataFrame()
            for pigfarm_dk in pigfarm_dks:
                config.EvalFilename.feature = f'{pigfarm_dk}'
                config.EvalFilename.version = version

                abortion_abnormal_eval = AbortionAbnormalPredictEval(logger)
                abortion_abnormal_eval.build_eval_set(start_date, 30, pigfarm_dk=pigfarm_dk, use_cache=True)
                data = abortion_abnormal_eval.eval_with_index_sample()
                data['pigfarm_dk'] = pigfarm_dk  # 添加猪场标识
                eval_result = pd.concat([eval_result, data], ignore_index=True)

            # eval_result.to_csv(f'{config.EvalFilename.version}_nfm_{config.EvalFilename.feature}.csv', index=False, encoding='utf-8', float_format="%.4f")
            all_result[version] = eval_result

    # 将all_result保存到一个csv文件中
    all_columns = ['pigfarm_dk', 'version'] + overall_col + special_col
    all_result_df = pd.DataFrame()
    # all_result_df = pd.concat([all_result_df, pd.DataFrame([{col: ' ' for col in all_columns}])], ignore_index=True)

    # 按照要求的顺序构建数据
    for i, pigfarm_dk in enumerate(pigfarm_dks):
        # 为每个猪场添加三个版本的数据，每个版本有三个eval_period
        eval_periods = ['1_7', '8_14', '15_21']

        for version in ['v1.0.r', 'v1.0.b', 'v1.0.t']:
            for eval_period in eval_periods:
                if version in all_result and not all_result[version].empty:
                    # 查找该版本中对应猪场和eval_period的数据
                    pigfarm_period_data = all_result[version][
                        (all_result[version]['pigfarm_dk'] == pigfarm_dk) &
                        (all_result[version]['eval_period'] == eval_period)
                    ]

                    if not pigfarm_period_data.empty:
                        # 选择需要的列，如果列不存在则用NaN填充
                        row_data = {}
                        for col in all_columns:
                            if col in pigfarm_period_data.columns:
                                row_data[col] = pigfarm_period_data[col].iloc[0]
                            elif col == 'version':
                                row_data[col] = version
                            else:
                                row_data[col] = None
                        all_result_df = pd.concat([all_result_df, pd.DataFrame([row_data])], ignore_index=True)
                    else:
                        # 没有对应的数据，添加空行
                        empty_row = {col: None for col in all_columns}
                        empty_row['version'] = version
                        empty_row['pigfarm_dk'] = pigfarm_dk
                        empty_row['eval_period'] = eval_period
                        all_result_df = pd.concat([all_result_df, pd.DataFrame([empty_row])], ignore_index=True)
                else:
                    # 版本不存在，添加空行
                    empty_row = {col: None for col in all_columns}
                    empty_row['version'] = version
                    empty_row['pigfarm_dk'] = pigfarm_dk
                    empty_row['eval_period'] = eval_period
                    all_result_df = pd.concat([all_result_df, pd.DataFrame([empty_row])], ignore_index=True)

        # 在每个猪场的所有数据后添加空行分隔
        empty_separator = {col: None for col in all_columns}
        all_result_df = pd.concat([all_result_df, pd.DataFrame([empty_separator])], ignore_index=True)

    # 计算all_result_df的平均值
    # 定义需要计算平均值的数值列
    numeric_cols = overall_col[4:] + special_col  # 排除stats_dt，因为它是日期

    # 按version, period分组计算平均值
    version_averages = []
    for version in ['v1.0.r', 'v1.0.b', 'v1.0.t']:
        for eval_period in ['1_7', '8_14', '15_21']:
            # 筛选当前版本和周期的数据，排除空行
            version_period_data = all_result_df[
                (all_result_df['version'] == version) &
                (all_result_df['eval_period'] == eval_period) &
                (all_result_df['pigfarm_dk'].notna())  # 排除空行（空行的pigfarm_dk为None）
            ]

            if not version_period_data.empty:
                # 计算平均值
                avg_row = {'version': version, 'eval_period': eval_period, 'pigfarm_dk': 'AVERAGE'}

                # 对数值列计算平均值
                for col in numeric_cols:
                    if col in version_period_data.columns:
                        # 只对非空值计算平均值
                        values = version_period_data[col].dropna()
                        if len(values) > 0:
                            avg_row[col] = values.mean()
                        else:
                            avg_row[col] = None
                    else:
                        avg_row[col] = None

                # 对于stats_dt列，取第一个非空值作为代表
                if 'stats_dt' in version_period_data.columns:
                    stats_dt_values = version_period_data['stats_dt'].dropna()
                    if len(stats_dt_values) > 0:
                        avg_row['stats_dt'] = stats_dt_values.iloc[0]
                    else:
                        avg_row['stats_dt'] = None

                version_averages.append(avg_row)
            else:
                # 如果没有数据，添加空的平均值行
                avg_row = {'version': version, 'eval_period': eval_period, 'pigfarm_dk': 'AVERAGE'}
                for col in all_columns:
                    if col not in ['version', 'eval_period', 'pigfarm_dk']:
                        avg_row[col] = None
                version_averages.append(avg_row)

    # 将平均值行添加到结果DataFrame中
    if version_averages:
        avg_df = pd.DataFrame(version_averages)
        all_result_df = pd.concat([all_result_df, avg_df], ignore_index=True)

    # 保存为CSV文件
    all_result_df.to_csv('v1.0.all_nfm_result.csv', index=False, encoding='utf-8', float_format="%.4f")
    logger.info("所有猪场对比结果已保存到 v1.0.all_nfm_result.csv")



if __name__ == "__main__":
    # per_pigfarm_evaluate()

    # index = pd.read_csv('pigfarm_dks.csv')
    # pigfarm_dks = index['pigfarm_dk'].unique().tolist()
    # pigfarm_dks = [dk.replace('/', '@') for dk in pigfarm_dks]

    # train_predict_data = pd.read_csv('data/predict/abort_abnormal/v1.0.t train/v1.0.t train 12/abort_abnormal.csv')
    # all_pigfarm_evaluate(version=['v1.0.t'], features=['train'], predict_data=train_predict_data, pigfarm_dks=pigfarm_dks)

    all_pigfarm_evaluate(version=['v1.0.49'], features=['abortion_abnormal_window_expand14'])


