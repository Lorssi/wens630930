import os
import sys
import logging

import numpy as np
import pandas as pd
from pandas.core.interchange.dataframe_protocol import DataFrame

import configs.base_config as base_config
import configs.pigfarm_risk_prediction_config as risk_config

from dataset.base_dataset import BaseDataSet

# 显示所有行
pd.set_option('display.max_rows', None)
# 设置value的显示长度为100，默认为50
pd.set_option('max_colwidth', 100)

base_dir = os.path.split(os.path.realpath(__file__))[0]
program = os.path.basename(sys.argv[0])
logger = logging.getLogger(program)


class RiskPredictionFeatureDataset(BaseDataSet):

    def __init__(self, **param):
        super().__init__(param)
        self.production_feature_data = pd.read_csv(base_config.FeatureData.PRODUCTION_FEATURE_DATA.value)
        self.check_feature_data = pd.read_csv(base_config.FeatureData.CHECK_FEATURE_DATA.value)
        self.death_confirm_feature_data = pd.read_csv(base_config.FeatureData.DEATH_CONFIRM_FEATURE_DATA.value)
        self.org_feature_data = pd.read_csv(base_config.FeatureData.ORG_FEATURE_DATA.value)
        self.date_feature_data = pd.read_csv(base_config.FeatureData.DATE_FEATURE_DATA.value)
        self.intro_feature_data = pd.read_csv(base_config.FeatureData.INTRO_FEATURE_DATA.value)
        self.sorrounding_feature_data = pd.read_csv(base_config.FeatureData.SORROUNDING_FEATURE_DATA.value)
        self.rule_baseline_feature_data = pd.read_csv(base_config.FeatureData.RULE_BASELINE_FEATURE_DATA.value)
        self.abnormal_boar_feature_data = pd.read_csv(base_config.FeatureData.ABNORMAL_BOAR_FEATURE_DATA.value)
        self.immune_feature_data = pd.read_csv(base_config.FeatureData.IMMUNE_FEATURE_DATA.value)

        self.file_name = None  # 文件名
        self.index_data = pd.DataFrame()  # 索引数据

        self.data = pd.DataFrame()

    def _check_data(self):
        pass

    def _preprocessing_data(self, input_dataset: pd.DataFrame):
        index_data = input_dataset
        index_data['stats_dt'] = pd.to_datetime(index_data['stats_dt'])

        self.index_data = index_data.copy()

    def _get_production_feature(self):
        """
        使用merge和向量化操作获取mean_prop特征，避免逐行处理
        """
        logger.info("获取production特征...")

        # 复制数据并确保类型正确
        index_data = self.index_data.copy()
        production_feature_data = self.production_feature_data.copy()

        # 转换日期类型和其他数据类型
        index_data['stats_dt'] = pd.to_datetime(index_data['stats_dt'])
        production_feature_data['stats_dt'] = pd.to_datetime(production_feature_data['stats_dt'])

        # production_feature_data['stats_dt'] = production_feature_data['stats_dt'] + pd.DateOffset(days=1)  # 向后偏移一天
        index_data = pd.merge(
            index_data,
            production_feature_data,
            on=['stats_dt', 'pigfarm_dk'],
            how='left'
        )

        index_data.rename(columns={'abortion_rate': 'abortion_rate_1_7'}, inplace=True)

        self.index_data = index_data.copy()


    def _get_date_feature(self):
        """
        使用merge和向量化操作获取mean_prop特征，避免逐行处理
        """
        logger.info("获取date特征...")

        # 复制数据并确保类型正确
        index_data = self.index_data.copy()
        date_feature_data = self.date_feature_data.copy()

        # 转换日期类型和其他数据类型
        index_data['stats_dt'] = pd.to_datetime(index_data['stats_dt'])
        date_feature_data['stats_dt'] = pd.to_datetime(date_feature_data['stats_dt'])

        index_data = pd.merge(
            index_data,
            date_feature_data,
            on=['stats_dt', 'pigfarm_dk'],
            how='left'
        )

        self.index_data = index_data.copy()


    def _get_org_feature(self):
        """
        使用merge和向量化操作获取mean_prop特征，避免逐行处理
        """
        logger.info("获取org特征...")

        # 复制数据并确保类型正确
        index_data = self.index_data.copy()
        org_feature_data = self.org_feature_data.copy()

        # 转换日期类型和其他数据类型
        index_data['stats_dt'] = pd.to_datetime(index_data['stats_dt'])

        index_data = pd.merge(
            index_data,
            org_feature_data,
            left_on=['pigfarm_dk'],
            right_on=['org_inv_dk'],
            how='left'
        ).drop(columns=['org_inv_dk'])

        self.index_data = index_data.copy()

    def _get_check_feature(self):
        """
        使用merge和向量化操作获取mean_prop特征，避免逐行处理
        """
        logger.info("获取check特征...")

        # 复制数据并确保类型正确
        index_data = self.index_data.copy()
        check_feature_data = self.check_feature_data.copy()

       # 转换日期类型和其他数据类型
        index_data['stats_dt'] = pd.to_datetime(index_data['stats_dt'])
        check_feature_data['stats_dt'] = pd.to_datetime(check_feature_data['stats_dt'])

        # check_feature_data['stats_dt'] = check_feature_data['stats_dt'] + pd.DateOffset(days=1)  # 向后偏移一天
        index_data = pd.merge(
            index_data,
            check_feature_data,
            on=['stats_dt', 'pigfarm_dk'],
            how='left'
        )

        self.index_data = index_data.copy()

    def _get_death_confirm_feature(self):
        """
        使用merge和向量化操作获取mean_prop特征，避免逐行处理
        """
        logger.info("获取check特征...")

        # 复制数据并确保类型正确
        index_data = self.index_data.copy()
        death_confirm_feature_data = self.death_confirm_feature_data.copy()

       # 转换日期类型和其他数据类型
        index_data['stats_dt'] = pd.to_datetime(index_data['stats_dt'])
        death_confirm_feature_data['stats_dt'] = pd.to_datetime(death_confirm_feature_data['stats_dt'])

        # death_confirm_feature_data['stats_dt'] = death_confirm_feature_data['stats_dt'] + pd.DateOffset(days=1)  # 向后偏移一天
        index_data = pd.merge(
            index_data,
            death_confirm_feature_data,
            on=['stats_dt', 'pigfarm_dk'],
            how='left'
        )

        self.index_data = index_data.copy()

    def _get_intro_feature(self):
        """
        使用merge和向量化操作获取mean_prop特征，避免逐行处理
        """
        logger.info("获取intro特征...")
        intro_feature = ['intro_source_num_90d', 'intro_source_is_single', 'intro_times_30d', 'intro_times_90d', 'intro_days_30d', 'intro_days_90d']

        # 复制数据并确保类型正确
        index_data = self.index_data.copy()
        intro_feature_data = self.intro_feature_data.copy()

       # 转换日期类型和其他数据类型
        index_data['stats_dt'] = pd.to_datetime(index_data['stats_dt'])
        intro_feature_data['stats_dt'] = pd.to_datetime(intro_feature_data['stats_dt'])

        # intro_feature_data['stats_dt'] = intro_feature_data['stats_dt'] + pd.DateOffset(days=1)  # 向后偏移一天
        index_data = pd.merge(
            index_data,
            intro_feature_data,
            on=['stats_dt', 'pigfarm_dk'],
            how='left'
        )

        self.index_data = index_data.copy()

    def _get_sorrounding_feature(self):
        """
        使用merge和向量化操作获取mean_prop特征，避免逐行处理
        """
        logger.info("获取intro特征...")
        sorrounding_feature = ['l3_abortion_mean', 'l3_abortion_mean_7d', 'l3_abortion_mean_15d', 'l3_abortion_mean_30d']

        # 复制数据并确保类型正确
        index_data = self.index_data.copy()
        sorrounding_feature_data = self.sorrounding_feature_data.copy()

       # 转换日期类型和其他数据类型
        index_data['stats_dt'] = pd.to_datetime(index_data['stats_dt'])
        sorrounding_feature_data['stats_dt'] = pd.to_datetime(sorrounding_feature_data['stats_dt'])
        index_data['pigfarm_dk'] = index_data['pigfarm_dk'].astype(str)

        # sorrounding_feature_data['stats_dt'] = sorrounding_feature_data['stats_dt'] + pd.DateOffset(days=1)  # 向后偏移一天
        index_data = pd.merge(
            index_data,
            sorrounding_feature_data,
            on=['stats_dt', 'pigfarm_dk'],
            how='left'
        )

        self.index_data = index_data.copy()

    def _get_rule_baseline_feature(self):
        """
        使用merge和向量化操作获取mean_prop特征，避免逐行处理
        """
        logger.info("获取rule baseline特征...")

        # 复制数据并确保类型正确
        index_data = self.index_data.copy()
        rule_baseline_feature_data = self.rule_baseline_feature_data.copy()

       # 转换日期类型和其他数据类型
        index_data['stats_dt'] = pd.to_datetime(index_data['stats_dt'])
        rule_baseline_feature_data['stats_dt'] = pd.to_datetime(rule_baseline_feature_data['stats_dt'])

        # rule_baseline_feature_data['stats_dt'] = rule_baseline_feature_data['stats_dt'] + pd.DateOffset(days=1)
        index_data = pd.merge(
            index_data,
            rule_baseline_feature_data,
            on=['stats_dt', 'pigfarm_dk'],
            how='left'
        )
        self.index_data = index_data.copy()

    def _get_abnormal_feature(self):
        """
        使用merge和向量化操作获取mean_prop特征，避免逐行处理
        """
        logger.info("获取abnormal boar特征...")

        # 复制数据并确保类型正确
        index_data = self.index_data.copy()
        abnormal_boar_feature_data = self.abnormal_boar_feature_data.copy()

       # 转换日期类型和其他数据类型
        index_data['stats_dt'] = pd.to_datetime(index_data['stats_dt'])
        abnormal_boar_feature_data['stats_dt'] = pd.to_datetime(abnormal_boar_feature_data['stats_dt'])

        # abnormal_boar_feature_data['stats_dt'] = abnormal_boar_feature_data['stats_dt'] + pd.DateOffset(days=1)  # 向后偏移一天
        index_data = pd.merge(
            index_data,
            abnormal_boar_feature_data,
            on=['stats_dt', 'pigfarm_dk'],
            how='left'
        )

        self.index_data = index_data.copy()

    # 免疫特征
    def _get_immune_feature(self):
        """
        使用merge和向量化操作获取mean_prop特征，避免逐行处理
        """
        logger.info("获取免疫特征...")

        # 复制数据并确保类型正确
        index_data = self.index_data.copy()
        immune_feature_data = self.immune_feature_data.copy()

       # 转换日期类型和其他数据类型
        index_data['stats_dt'] = pd.to_datetime(index_data['stats_dt'])
        immune_feature_data['stats_dt'] = pd.to_datetime(immune_feature_data['stats_dt'])

        # immune_feature_data['stats_dt'] = immune_feature_data['stats_dt'] + pd.DateOffset(days=1)  # 向后偏移一天
        index_data = pd.merge(
            index_data,
            immune_feature_data,
            on=['stats_dt', 'pigfarm_dk'],
            how='left'
        )

        self.index_data = index_data.copy()

    def _post_processing_train_data(self):
        index_data = self.index_data.copy()

        # 统计各字段nan值
        null_counts = index_data.isnull().sum()
        print(null_counts)
        logger.info("-----Processed Dataset Size：{}".format(len(index_data)))

        self.file_name = "risk_train_connected_feature_data.csv"

        self.data = index_data.copy()

    def _post_processing_predict_data(self):
        index_data = self.index_data.copy()
        # 统计各字段nan值
        null_counts = index_data.isnull().sum()
        print(null_counts)
        logger.info("-----Processed Dataset Size：{}".format(len(index_data)))

        self.file_name = "test_connected_feature_data.csv"
        self.data = index_data.copy()
        

    def build_train_dataset(self, input_dataset: pd.DataFrame, **param):
        logger.info("-----Checking Data-----")
        self._check_data()
        logger.info("-----Preprocessing Data")
        self._preprocessing_data(input_dataset=input_dataset)
        logger.info("-----Connecting feature: mean_prop fea")
        self._get_production_feature()
        logger.info("-----Connecting feature: prev_prop fea")
        self._get_org_feature()
        logger.info("-----Connecting feature: cur_month fea")
        self._get_date_feature()
        logger.info("-----Connecting feature: cur_days fea")
        self._get_check_feature()
        logger.info("-----Connecting feature: correct_status_nm fea")
        self._get_death_confirm_feature()
        logger.info("-----Connecting feature: intro_data")
        # self._get_intro_feature()
        logger.info("-----Connecting feature: Sorrounding fea")
        # self._get_sorrounding_feature()
        logger.info("-----Connecting feature: rule baseline fea")
        self._get_rule_baseline_feature()
        logger.info("-----Connecting feature: abnormal boar fea")
        self._get_abnormal_feature()
        logger.info("-----Postprocessing Data-----")
        self._post_processing_train_data()
        self.dump_dataset(risk_config.algo_interim_dir / self.file_name)
        logger.info("-----Done----- ")
        return self.data

    def build_predict_dataset(self, input_dataset: pd.DataFrame, **param):
        logger.info("-----Checking Data-----")
        self._check_data()
        logger.info("-----Preprocessing Data")
        self._preprocessing_data(input_dataset=input_dataset)
        logger.info("-----Connecting feature: mean_prop fea")
        self._get_mean_prop_feature()
        logger.info("-----Connecting feature: prev_prop fea")
        self._get_prev_prop_feature()
        logger.info("-----Connecting feature: cur_month fea")
        self._get_cur_month_feature()
        logger.info("-----Connecting feature: cur_days fea")
        self._get_cur_days_feature()
        logger.info("-----Postprocessing Data-----")
        self._post_processing_predict_data()
        # self.dump_dataset("/".join([prop_config.Prediction_interim_dir, self.file_name]))
        logger.info("-----Done----- ")
        return self.data



if __name__ == '__main__':
    dataset = PropPredictionFeatureDataset()
    df1 = pd.read_csv(r"C:\Users\636\Desktop\Poultry-Performance-Prediction-Pricing\data\interim\Performance_Prediction\prop\train_index_sample_data.csv")
    df2 = pd.read_csv(r"C:\Users\636\Desktop\Poultry-Performance-Prediction-Pricing\data\interim\Performance_Prediction\prop\test_index_sample_data.csv")
    dataset.build_train_dataset(df1)
    dataset.build_predict_dataset(df2)
