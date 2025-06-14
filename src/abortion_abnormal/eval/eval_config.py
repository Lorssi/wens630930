from enum import Enum
import os

# 中间数据目录
INTERIM_DIR = "data/interim_data"

# 特征名称，用于特征消融
class EvalFilename():
    version = None # 版本号
    feature = None # 特征文件名
    eval_date = None # 评估开始日期

# 评估目录
eval_dir = os.path.join(INTERIM_DIR, 'eval')
os.makedirs(eval_dir, exist_ok=True)

# 任务1 流产率异常预测评估数据保存路径
abortion_abnormal_eval_save_path = os.path.join(eval_dir, 'abortion_abnormal_eval_data')
os.makedirs(abortion_abnormal_eval_save_path, exist_ok=True)
abortion_abnormal_eval_index_sample_save_path = os.path.join(abortion_abnormal_eval_save_path, 'abortion_abnormal_index_sample.csv')
abortion_abnormal_eval_ground_truth_save_path = os.path.join(abortion_abnormal_eval_save_path, 'abortion_abnormal_ground_truth.csv')
abortion_abnormal_eval_result_save_path = os.path.join(abortion_abnormal_eval_save_path, '{}_abortion_abnormal_{}.xlsx')

# 任务2 流产率天数预测评估数据保存路径
abortion_days_eval_save_path = os.path.join(eval_dir, 'abortion_day_eval_data')
os.makedirs(abortion_days_eval_save_path, exist_ok=True)
abortion_days_eval_index_sample_save_path = os.path.join(abortion_days_eval_save_path, 'abortion_day_index_sample.csv')
abortion_days_eval_ground_truth_save_path = os.path.join(abortion_days_eval_save_path, 'abortion_day_ground_truth.csv')
abortion_days_eval_result_save_path = os.path.join(abortion_days_eval_save_path, '{}_abortion_days_{}.xlsx')

# delete
class ModulePath(Enum):
    eval_list = [
    # 任务1 流产率异常预测
        # {
        #   'eval_class_name': 'abortion_abnormal.eval.main',
        #   'eval_main_class_name': 'AbortionAbnormalPredictEval',
        #   'params': {}
        # },
    # 任务2 流产率天数预测
        {
          'eval_class_name': 'abortion_abnormal.eval.main',
          'eval_main_class_name': 'AbortionDayPredictEval',
          'params': {}
        },
    ]