import importlib
import logging
import os
import sys

import configs.base_config as config


class TrainModelMain(object):

    def __init__(self, **param):
        self.train_algo_outcome_dt_end = param.get('train_algo_outcome_dt_end')
        self.logger = param.get('logger')
        self.train_algo_arr = param.get('train_algo_arr')


    def train_model_generality(self, algo_name: str,
                               algo_main_class: str,
                               alog_config_name: str,
                               algo_simple_name: str):
        # 判断是否执行该训练任务
        if algo_simple_name in self.train_algo_arr:
            self.logger.info('----------------------------------------开始训练模型， 任务： %s ----------------------------------------' % algo_name)
            algo_module = importlib.import_module(algo_name)
            algo_config = importlib.import_module(alog_config_name)
            main_class = getattr(algo_module, algo_main_class)
            main_class_instance = main_class()
            # valid_algo_running_dt_end = None
            # valid_algo_interval = None
            main_class_instance.train(self.train_algo_outcome_dt_end, algo_config.TrainModuleConfig.TRAIN_ALGO_INTERVAL.value, param={})
            self.logger.info('----------------------------------------任务： %s 模型训练完成----------------------------------------' % algo_name)

    def train_all_model(self):
        self.logger.info("----------------------------------------开始训练模型，时间: %s ----------------------------------------" % self.train_algo_outcome_dt_end)
        algo_list = config.ModulePath.algo_list.value
        for algo_module in algo_list:
            self.train_model_generality(algo_module['algo_name'], algo_module['algo_main_class'],
                                        algo_module['algo_config'], algo_module['algo_simple_name'])
        self.logger.info("----------------------------------------模型生成完成，进入预测模块----------------------------------------")