import os


INTERIM_DIR = "data/interim_data"

# 评估目录
eval_dir = os.path.join(INTERIM_DIR, 'eval')
os.makedirs(eval_dir, exist_ok=True)

# 流产率异常预测评估数据保存路径
abortion_abnormal_eval_save_path = os.path.join(eval_dir, 'abortion_abnormal_eval_data')
os.makedirs(abortion_abnormal_eval_save_path, exist_ok=True)
abortion_abnormal_eval_index_sample_save_path = os.path.join(abortion_abnormal_eval_save_path, 'abortion_abnormal_index_sample.csv')
abortion_abnormal_eval_ground_truth_save_path = os.path.join(abortion_abnormal_eval_save_path, 'abortion_abnormal_ground_truth.csv')
abortion_abnormal_eval_result_save_path = os.path.join(abortion_abnormal_eval_save_path, '{}_abortion_abnormal_{}.xlsx')

# 流产率天数预测评估数据保存路径
abortion_days_eval_save_path = os.path.join(eval_dir, 'abortion_day_eval_data')
os.makedirs(abortion_days_eval_save_path, exist_ok=True)
abortion_days_eval_index_sample_save_path = os.path.join(abortion_days_eval_save_path, 'abortion_day_index_sample.csv')
abortion_days_eval_ground_truth_save_path = os.path.join(abortion_days_eval_save_path, 'abortion_day_ground_truth.csv')
abortion_days_eval_result_save_path = os.path.join(abortion_days_eval_save_path, '{}_abortion_days_{}.xlsx')