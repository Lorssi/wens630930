import argparse
import importlib
import logging
import os
import sys
from configs.base_config import ModulePath



class FeatureGenerateMain(object):

    def __init__(self, **param):
        self.running_dt = param.get('running_dt')
        self.origin_feature_precompute_interval = param.get('origin_feature_precompute_interval')
        self.logger = param.get('logger')

    def origin_feature_pre_generate_all(self, dataset_name, main_class_name: str, running_dt: str,origin_feature_precompute_interval: int,
                                        file_type: str,
                                        params):
        self.logger.info('正在生成 %s 特征 ' % main_class_name)
        dataset_module = importlib.import_module(dataset_name)
        main_class = getattr(dataset_module, main_class_name)
        main_class_instance = main_class(running_dt,origin_feature_precompute_interval, file_type, **params)
        main_class_instance.build_dataset_all()

    def origin_feature_pre_generate_add(self, class_name, running_dt: str, origin_feature_precompute_interval: int, params: dict):
        obj = globals()[class_name]
        obj.build_dataset_add(running_dt, origin_feature_precompute_interval, **params)

    def generate_feature(self):
        self.logger.info("开始生成特征，开始时间 %s " % (self.running_dt))
        feature_dataset_list = ModulePath.feature_dataset_list.value
        for feature_dataset in feature_dataset_list:
            self.origin_feature_pre_generate_all(feature_dataset['dataset_name'], feature_dataset['main_class_name'],
                                                 self.running_dt, self.origin_feature_precompute_interval,feature_dataset['file_type'],
                                                 feature_dataset['params'])
        self.logger.info('特征模块成功完成，进入模型训练模块')


