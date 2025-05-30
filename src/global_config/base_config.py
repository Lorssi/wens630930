from enum import Enum
from utils import os_util
import socket
import warnings

class ModulePath(Enum):
    # 数据预处理模块
    pre_process_data_class = 'PreProcessDataset'
    # 特征模块
    feature_dataset_list = [
        {'dataset_name': 'feature_alter.train_index_dataset',
         'main_class_name': 'TrainIndexDataSet',  # xx.py
         'file_type': 'csv',
         'params': {}},
    ]

# todo 任务模块定义
class TaskModule(Enum):
    eval_task_list = [
        {
            'task_name': 'abortion_abnormal.abortion_abnormal_evaluator',  # xx.py
            'task_main_class': 'AbortionAbnormalEvaluator',  # xx.py下面的class
            'task_simple_name': 'abortion_abnormal_predict'  # 任务名称
        },
    ]