import pandas as pd
import logging
import os
import sys
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, classification_report
from sklearn.exceptions import UndefinedMetricWarning
from tqdm import tqdm
import warnings

from abortion_abnormal.eval.eval_base import EvalBaseMixin
from abortion_abnormal.eval.build_eval_dateset import abortion_abnormal_index_sample
from abortion_abnormal.eval.evaluation_test import EvalData

# base_dir = os.path.split(os.path.realpath(__file__))[0]
program = os.path.basename(sys.argv[0])
logger = logging.getLogger(program)
logging.basicConfig(format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s: %(message)s',
                    datefmt='%a, %d %b %Y %H:%M:%S')
logging.root.setLevel(level=logging.INFO)
# warnings.filterwarnings("ignore")

# 任务1评估
class AbortionAbnormalEval1(EvalBaseMixin):
    def __init__(self, logger=None):
        self.logger = logger
        self.result = pd.DataFrame()

    # 构建评测数据集
    def build_eval_set(self, eval_running_dt, eval_interval, param=None):
        eval_running_dt = pd.to_datetime(eval_running_dt)
        self.eval_running_dt_start = (eval_running_dt + pd.Timedelta(days=1)).strftime('%Y-%m-%d')
        self.eval_running_dt_end = (eval_running_dt + pd.Timedelta(days=eval_interval)).strftime('%Y-%m-%d')

        # 获取真实值
        self.sample_index, self.sample_ground_truth = abortion_abnormal_index_sample(
            eval_running_dt_start=self.eval_running_dt_start,
            eval_running_dt_end=self.eval_running_dt_end
        )

        # todo 获取预测值
        self.predict_data = pd.read_csv("data/predict/has_risk_predict_result.csv", encoding='utf-8')
        # todo 转化日期格式
        self.predict_data['stats_dt'] = pd.to_datetime(self.predict_data['stats_dt'])

        # todo 
        self.predict_data = self.predict_data.drop(columns=['abort_1_7', 'abort_8_14', 'abort_15_21'])

        # 获取组织分类表
        dim_org_inv = pd.read_csv("data/raw_data/dim_org_inv.csv", encoding='utf-8')

        # 处理组织分类信息，只保留需要的列
        org_mapping = dim_org_inv[['org_inv_dk', 'l2_org_inv_nm', 'l3_org_inv_nm', 'l4_org_inv_nm']].copy()

        # 确保ID列为字符串类型以避免join时的类型不匹配问题
        org_mapping['org_inv_dk'] = org_mapping['org_inv_dk'].astype(str)
        self.predict_data['pigfarm_dk'] = self.predict_data['pigfarm_dk'].astype(str)
        
        # 将与组织数据合并，添加部门信息
        self.sample_ground_truth = self.sample_ground_truth.merge(
            org_mapping,
            left_on='pigfarm_dk',
            right_on='org_inv_dk',
            how='left'
        )



    def get_eval_index_sample(self):
        return self.index_sample


    # def eval_with_index_sample(self, predict_result, save_flag=True):
    def eval_with_index_sample(self, save_flag=True):
        eval_dt = EvalData(self.predict_data, self.sample_ground_truth, self.eval_running_dt_start, self.eval_running_dt_end, logger)

        all_sample_result = eval_dt.eval_all_samples()
        all_sample_result_exclude_feiwen = eval_dt.eval_all_samples_exclude_feiwen()
        special_samples_result = eval_dt.eval_special_samples()
        l2_results, l3_results, l4_results = eval_dt.eval_organizational_hierarchy()

        if save_flag:
            with pd.ExcelWriter(f"data/eval/abortion_abnormal_eval_{self.eval_running_dt_end}.xlsx") as writer:
                all_sample_result.to_excel(writer, sheet_name='整体', index=False)
                all_sample_result_exclude_feiwen.to_excel(writer, sheet_name='整体-剔除非瘟数据', index=False)
                special_samples_result.to_excel(writer, sheet_name='特殊样本分析', index=False)
                l2_results.to_excel(writer, sheet_name='二级组织分类', index=False)
                l3_results.to_excel(writer, sheet_name='三级组织分类', index=False)
                l4_results.to_excel(writer, sheet_name='四级组织分类', index=False)


# 任务2评估
class AbortionAbnormalEval2(EvalBaseMixin):
    pass

# 任务3评估
class AbortionAbnormalEval3(EvalBaseMixin):
    pass