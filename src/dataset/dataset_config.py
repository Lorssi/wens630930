import socket

import pandas as pd

from util import os_util


"""
此为例子，每个训练算法或相关联的训练算法所需的「参数配置」区，例如涉及以下部件的相关参数
1. 样本划分，例如训练数据时间范围，测试数据时间范围
2. 特征处理模块配套参数
3. 训练缓解涉及的采样参数，例如最大负采样数量
4. 中间过程涉及的目录地址，临时文件名等

以下为范例
"""

local_ip = socket.gethostbyname(socket.gethostname())
if local_ip == '10.11.41.17':
    base_dir = "F:/scau/precisionMarketing/项目代码学习/pig_org_diarrhea_predict"  # 生产机器
else:
    base_dir = '..'

RAW_DATA_ROOT = os_util.create_dir_if_not_exist('/'.join([base_dir, 'data/raw']))
INTERIM_DATA_ROOT = os_util.create_dir_if_not_exist('/'.join([base_dir, 'data/interim']))
external_dir = os_util.create_dir_if_not_exist('/'.join([base_dir, 'data/external']))
model_dir = os_util.create_dir_if_not_exist('/'.join([base_dir, 'data/model']))

raw_task2_lightgbm_dir = os_util.create_dir_if_not_exist('/'.join([base_dir, 'data/raw/task2_lightgbm_raw']))

MODEL_TASK_TWO_ROOT = os_util.create_dir_if_not_exist('/'.join([base_dir, 'data/interim/task2_lightgbm_interim']))

external_task2_lightgbm_dir = os_util.create_dir_if_not_exist('/'.join([base_dir, 'data/external/task2_lightgbm_model_predict']))

model_task2_lightgbm_dir = os_util.create_dir_if_not_exist('/'.join([base_dir, 'data/model/task2_lightgbm_model_train']))

log_dir = os_util.create_dir_if_not_exist('/'.join([base_dir, 'data/model/train_log']))

