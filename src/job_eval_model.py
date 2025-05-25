import importlib
from global_config import base_config as global_config
import pandas as pd
import argparse
import logging
from utils.logger import init_logger
import sys
import os
from datetime import datetime, timedelta

eval_path = global_config.JOB_EVAL_LOG_PATH
program = os.path.basename(sys.argv[0])
logger = logging.getLogger(program)
logging.basicConfig(format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s: %(message)s', datefmt='%a, %d %b %Y %H:%M:%S')
logger = init_logger(eval_path)


def str_to_list(s):
    return s.split(',')


class EVALTaskMain(object):
    def __init__(self, **param):
        self.eval_task_arr = param.get('eval_task_arr')
        self.all_task_param = param.get('all_task_param')

    def task_eval_generality_post_process(self, task_name: str,
                                            task_main_class: str,
                                            task_simple_name: str,):

        # 判断是否要执行某个任务
        logger.info("self.eval_task_arr:{}".format(self.eval_task_arr))

        if task_simple_name in self.eval_task_arr:
            logger.info("----------------------------------------任务: %s 开始预测评估流程数据----------------------------------------" % task_simple_name)
            try:
                algo_module = importlib.import_module(task_name)
                main_class = getattr(algo_module, task_main_class)
                main_class_instance = main_class(task_param=self.all_task_param)
                eval_result = main_class_instance.eval_and_post_process()
                return eval_result
            
            except Exception as e:
                return e

    def task_eval_post_process(self):
        task_list = global_config.TaskModule.eval_task_list.value
        for task_module in task_list:
            train_result = self.task_eval_generality_post_process(
                task_module['task_name'],
                task_module['task_main_class'],
                task_module['task_simple_name'],
            )
            logger.info(train_result)
        self.post_process()

    # 后处理
    def post_process(self):
        logger.info("正在执行评估主流程 后处理")
        logger.info("正在执行评估主流程 后处理 完成")



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='This is a simple command line tool.')
    logger.info('----------------------------------------初始化参数----------------------------------------')

    # 预警
    parser.add_argument('--eval_running_dt_end', default="2024-03-26", type=str, help='Input eval running dt end')
    parser.add_argument('--predict_interval', default=21, type=int, help='predict_interval')

    # 统一 todo 将default改为流产率预测类
    parser.add_argument('--eval_task_arr', type=str_to_list , default='abortion_abnormal_predict', help='Input train task dk')

    # 参数初始化
    args = parser.parse_args()
    eval_task_arr = args.eval_task_arr
    logger.info("执行评估的任务：%s"%eval_task_arr)


    # # 任务参数
    all_task_param = {
        # 预警
        "eval_running_dt_end": args.eval_running_dt_end,
        "predict_interval":args.predict_interval,
        # 统一
        "eval_task_arr": eval_task_arr
    }


    logger.info("all_task_param:{}".format(all_task_param))
    train_task_main = EVALTaskMain(eval_task_arr=eval_task_arr, all_task_param=all_task_param)
    train_task_main.task_eval_post_process()

