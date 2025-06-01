"""
_____________________________
  Author: 叶双福
   Time : 2023/9/5
   File : data_pre_process_main.py
______________________________
"""
import sys

import pandas as pd

import configs.base_config as config


class DataPreProcessMain(object):

    def __init__(self, **param):
        self.production_info = None
        self.org_info = None
        self.death_confirm_info = None
        self.check_info = None

        self.logger = param.get('logger')

    def get_dataset(self):
        self.production_info = pd.read_csv(config.RawData.ADS_PIG_ORG_TOTAL_TO_ML_TRAINING_DAY.value, encoding='utf-8')
        self.org_info = pd.read_csv(config.RawData.DIM_ORG_INV.value, encoding='utf-8')
        self.death_confirm_info = pd.read_csv(config.RawData.TMP_ORG_PRRS_OVERALL_ADOPT_CULL_DAY.value, encoding='utf-8')
        self.check_info = pd.read_csv(config.RawData.TMP_PIG_ORG_DISEASE_CHECK_RESULT_DAY.value, encoding='utf-8')

    def check_columns(self, df_columns: list, data_dict_columns: list):
        # 检查列名是否一致
        if df_columns == data_dict_columns:
            self.logger.info('同步拉取的表的列与数据字典的列一致')
        else:
            self.logger.info("列名不一致, 中止程序")
            sys.exit(1)

    def check_production_info_data(self):
        # 检测列名是否一致
        df_columns = self.production_info.columns
        pigfarm_dk_is_null = self.production_info["pigfarm_dk"].isnull().any()
        if pigfarm_dk_is_null:
            self.logger.error('猪场id空值')
            # sys.exit(1)
        else:
            self.logger.info('----------生产数据无异常----------')

    def check_org_info_data(self):
        org_info_data_duplicates = self.org_info[
            self.org_info.duplicated(['l4_org_inv_dk', 'org_inv_dk'])]

        if not org_info_data_duplicates.empty:
            self.logger.error('组织维度表id存在重复值')
        else:
            self.logger.info('----------组织维度表数据无异常----------')


    def check_death_info_data(self):
        self.logger.info('----------死淘表数据无异常----------')

    def check_check_info_data(self):
        self.logger.info('----------检测表数据无异常----------')


    def data_post_process(self):
        pass

    def check_data(self):

        self.logger.info('----------开始检查数据----------')
        self.get_dataset()
        self.logger.info('----------检查生产数据----------')
        self.check_production_info_data()
        self.logger.info('----------检查组织数据----------')
        self.check_org_info_data()
        self.logger.info('----------检查死淘数据----------')
        self.check_death_info_data()
        self.logger.info('----------检查检测数据----------')
        self.check_check_info_data()

        self.data_post_process()
        self.logger.info('----------同步数据检测无异常----------')
