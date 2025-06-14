import pandas as pd
import logging
import os
import sys
from pathlib import Path

# 获取项目根目录（假设是 src 的父目录）
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_root))

from abortion_abnormal.eval.eval_base import EvalBaseMixin
from abortion_abnormal.eval.build_eval_dateset import abortion_abnormal_index_sample, abortion_days_index_sample
from abortion_abnormal.eval.evaluation_test import *
import abortion_abnormal.eval.eval_config as config

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
        self.eval_running_dt_start = None
        self.eval_running_dt_end = None

    # 构建评测数据集
    def build_eval_set(self, eval_running_dt, eval_interval, param=None):
        eval_running_dt = pd.to_datetime(eval_running_dt)
        self.eval_running_dt_start = (eval_running_dt + pd.Timedelta(days=1)).strftime('%Y-%m-%d')
        self.eval_running_dt_end = (eval_running_dt + pd.Timedelta(days=eval_interval)).strftime('%Y-%m-%d')

        # 获取真实值
        self.index_sample, self.sample_ground_truth = abortion_abnormal_index_sample(self.eval_running_dt_start, self.eval_running_dt_end)
        self.index_sample.to_csv(config.abortion_abnormal_eval_index_sample_save_path, index=False, encoding='utf-8')
        self.sample_ground_truth.to_csv(config.abortion_abnormal_eval_ground_truth_save_path, index=False, encoding='utf-8')

        # delete 获取预测结果
        feature = config.EvalFilename.feature
        if feature is None:
            predict_file_path = "data/predict/abort_abnormal/abort_abnormal.csv"
        else:
            predict_file_path = f"data/predict/abort_abnormal/{config.EvalFilename.version} {feature}/{config.EvalFilename.version} {feature} {config.EvalFilename.eval_date}/abort_abnormal.csv"
        self.predict_data = pd.read_csv(predict_file_path, encoding='utf-8')

        # delete 转化日期格式
        self.predict_data['stats_dt'] = pd.to_datetime(self.predict_data['stats_dt'])

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

        # tmp = self.sample_ground_truth.merge(
        #     self.predict_data,
        #     on=['pigfarm_dk', 'stats_dt'],
        #     how='left'
        # )

        # tmp.to_csv('tmp.csv', index=False, encoding='utf-8')
        # assert False


    def get_eval_index_sample(self):
        return self.index_sample


    # def eval_with_index_sample(self, predict_result, save_flag=True):
    def eval_with_index_sample(self, predict_data=None, save_flag=True):
        eval_dt = AbortionAbnormalPredictEvalData(self.predict_data, self.sample_ground_truth, self.eval_running_dt_start, self.eval_running_dt_end, logger)

        # 整体评估
        all_sample_result = eval_dt.eval_all_samples()
        # 整体评估-剔除非瘟数据
        all_sample_result_exclude_feiwen = eval_dt.eval_all_samples_exclude_feiwen()
        # 特殊样本分析
        special_samples_result = eval_dt.eval_special_samples()
        # 分级组织评估
        l2_results, l3_results, l4_results = eval_dt.eval_organizational_hierarchy()
        # 流产持续时长统计
        abortion_duration_results = eval_dt.calculate_abortion_duration()
        # 流产间隔统计
        abortion_interval_results = eval_dt.calculate_abortion_interval()

        # delete 临时组合，复制方便
        data = all_sample_result_exclude_feiwen.copy()
        # 将specail_samples_result[sample_type]=='normal-2_to_abnormal-2'的precision,recall,f1_score,auc,special_recall列加在data后面
        normal_to_abnormal = special_samples_result[special_samples_result['sample_type'] == 'normal-2_to_abnormal-2'].reset_index().copy()
        if not normal_to_abnormal.empty:
            for col in ['precision', 'recall', 'f1_score', 'auc', 'special_recall']:
                if col in normal_to_abnormal.columns:
                    data[f'normal_to_abnormal_{col}'] = normal_to_abnormal[col]
        # 将specail_samples_result[sample_type]=='abnormal-2_to_normal-2'的precision,recall,f1_score,auc,special_recall列加在data后面
        abnormal_to_normal = special_samples_result[special_samples_result['sample_type'] == 'abnormal-2_to_normal-2'].reset_index().copy()
        if not abnormal_to_normal.empty:
            for col in ['precision', 'recall', 'f1_score', 'auc', 'special_recall']:
                if col in abnormal_to_normal.columns:
                    print('abnormal_to_normal:')
                    print(abnormal_to_normal[col])
                    data[f'abnormal_to_normal_{col}'] = abnormal_to_normal[col]
        # 将specail_samples_result[sample_type]=='abnormal_to_abnormal'的precision,recall,f1_score,auc,special_recall列加在data后面
        abnormal_to_abnormal = special_samples_result[special_samples_result['sample_type'] == 'abnormal_to_abnormal'].reset_index().copy()
        if not abnormal_to_abnormal.empty:
            for col in ['precision', 'recall', 'f1_score', 'auc', 'special_recall']:
                if col in abnormal_to_abnormal.columns:
                    data[f'abnormal_to_abnormal_{col}'] = abnormal_to_abnormal[col]
        # 将l2_results[l2_name]=='猪业一部'的stats_dt,l2_name,total_sample_num,remain_sample_num,eval_period,precision,recall,f1_score,auc,recognition列加在data后面
        pig_dept = l2_results[l2_results['l2_name'] == '猪业一部'].reset_index().copy()
        if not pig_dept.empty:
            cols_to_add = ['stats_dt', 'l2_name', 'total_sample_num', 'remain_sample_num', 
                          'eval_period', 'precision', 'recall', 'f1_score', 'auc', 'recognition']
            for col in cols_to_add:
                if col in pig_dept.columns:
                    data[f'pig_dept_{col}'] = pig_dept[col]
        data.to_csv(f'{config.EvalFilename.version}_{config.EvalFilename.feature}_{config.EvalFilename.eval_date}.csv', index=False, encoding='utf-8', float_format="%.4f")


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
                l2_results.to_excel(writer, sheet_name='二级组织分类', index=False, float_format="%.4f")
                l3_results.to_excel(writer, sheet_name='三级组织分类', index=False, float_format="%.4f")
                l4_results.to_excel(writer, sheet_name='四级组织分类', index=False, float_format="%.4f")
                abortion_duration_results.to_excel(writer, sheet_name='流产持续时长统计', index=False, float_format="%.4f")
                abortion_interval_results.to_excel(writer, sheet_name='流产间隔统计', index=False, float_format="%.4f")


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
            predict_file_path = f"data/predict/abort_days/{feature}/abort_abnormal_day.csv"
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
        all_sample_result_exclude_feiwen = eval_dt.eval_all_samples_exclude_feiwen()
        # 特殊样本分析
        special_samples_result = eval_dt.eval_special_samples()
        # 分级组织评估
        l2_results, l3_results, l4_results = eval_dt.eval_organizational_hierarchy()
        # 流产持续时长统计
        abortion_duration_results = eval_dt.calculate_abortion_duration()
        # 流产间隔统计
        abortion_interval_results = eval_dt.calculate_abortion_interval()
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


# 任务3评估-流产率预测评估
class AbortionAbnormalEval(EvalBaseMixin):
    pass

if __name__ == "__main__":
    print(project_root)

    features = [
        # 'preg_stock_sqty_change_ratio_7d', 'preg_stock_sqty_change_ratio_15d', 
        'reserve_sow_sqty_change_ratio_7d', 'reserve_sow_sqty_change_ratio_15d', 
        'basesow_sqty_change_ratio_7d', 'basesow_sqty_change_ratio_15d'
        ]


    version = ['v1.0.12', 'v1.0.13', 'v1.0.14', 'v1.0.15']

    for feature, version in zip(features, version):
        config.EvalFilename.feature = feature
        config.EvalFilename.version = version
        for (start_date, month) in [
            ('2024-02-29', '3'),
            ('2024-05-31', '6'),
            ('2024-08-31', '9'),
            ('2024-11-30', '12'),
        ]:
            # 任务1 流产率异常预测评估
            config.EvalFilename.eval_date = month
            abortion_abnormal_eval = AbortionAbnormalPredictEval(logger)
            abortion_abnormal_eval.build_eval_set(start_date, 30)
            abortion_abnormal_eval.eval_with_index_sample()