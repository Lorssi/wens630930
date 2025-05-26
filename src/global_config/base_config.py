from enum import Enum
from utils import os_util
import socket
import warnings

warnings.filterwarnings('ignore') # 关闭所有警告

# 设置base_dir
local_ip = socket.gethostbyname(socket.gethostname())
if local_ip == '10.11.41.17':
    base_dir = "/data0/hy_data_mining/pig_org_asf_predict_attribution/"  # 测试
elif local_ip == '10.11.21.201':
    base_dir = "/data0/hy_data_mining/pig_org_asf_predict_attribution/"  # 生产机器
else:
    base_dir = "../"

# todo 日志路径
LOG_ROOT = os_util.create_dir_if_not_exist('/'.join([base_dir, 'log']))

# 评估
JOB_EVAL_LOG_PATH = "/".join([LOG_ROOT, 'job_eval'])

# todo 任务模块定义
class TaskModule(Enum):
    eval_task_list = [
        {
            'task_name': 'abortion_abnormal.abortion_abnormal_evaluator',  # xx.py
            'task_main_class': 'AbortionAbnormalEvaluator',  # xx.py下面的class
            'task_simple_name': 'abortion_abnormal_predict'  # 任务名称
        },
    ]