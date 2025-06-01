import os
import sys
import logging
from operator import index

import numpy as np
import pandas as pd

from dataset.base_dataset import BaseDataSet
from datetime import datetime, timedelta
import configs.base_config as config
import configs.pigfarm_risk_prediction_config as risk_config

# 显示所有行
pd.set_option('display.max_rows', None)
# 设置value的显示长度为100，默认为50
pd.set_option('display.max_colwidth', 100)

base_dir = os.path.split(os.path.realpath(__file__))[0]
program = os.path.basename(sys.argv[0])
logger = logging.getLogger(program)


class RiskPredictionIndexSampleDataset(BaseDataSet):

    def __init__(self, **param):
        super().__init__(param)
        self.df_self = None
        self.df_breed = None
        self.data = None

    def _check_data(self):
        pass

    def _generate_samples(self):
        pass

    # 生成训练样本
    def _preprocessing_train_data(self, train_algo_outcome_dt_end, train_algo_interval):
        """
        预处理训练数据

        参数:
        train_algo_outcome_dt_end: 训练数据结束日期
        train_algo_interval: 训练数据时间间隔(天)
        """
        self.df_self = pd.read_csv(config.RawData.ADS_PIG_ORG_TOTAL_TO_ML_TRAINING_DAY.value)

        # 转换日期格式
        if not pd.api.types.is_datetime64_dtype(train_algo_outcome_dt_end):
            train_algo_outcome_dt_end = pd.to_datetime(train_algo_outcome_dt_end)

        # 过滤训练数据时间范围
        train_algo_outcome_dt_end = pd.to_datetime(train_algo_outcome_dt_end) - pd.Timedelta(days=1)
        train_algo_outcome_dt_start = train_algo_outcome_dt_end - pd.Timedelta(days=train_algo_interval)
        self.df_self['stats_dt'] = pd.to_datetime(self.df_self['stats_dt'])

        df_filtered = self.df_self[(self.df_self['stats_dt'] >= train_algo_outcome_dt_start) &
                                   (self.df_self['stats_dt'] <= train_algo_outcome_dt_end)]

        if df_filtered.empty:
            logger.warning("过滤后没有符合条件的训练数据")
            self.data = pd.DataFrame()
        else:
            logger.info(f"预处理后的数据量: {len(df_filtered)}")
            self.data = df_filtered

    def _postprocessing_train_data(self):
        """
        对生成的训练样本进行后处理，确保只保留所需特征
        """
        if self.data is None or self.data.empty:
            logger.warning("没有训练数据需要后处理")
            self.train_data = pd.DataFrame()
            return

        # 确保所有必要的列都存在
        required_features = ['stats_dt', 'pigfarm_dk']

        missing_features = [col for col in required_features if col not in self.data.columns]

        if missing_features:
            raise ValueError(f"数据缺少必要特征: {', '.join(missing_features)}")

        # 只保留所需特征列
        final_data = self.data[required_features].copy()

        # 确保特征类型正确
        final_data['stats_dt'] = pd.to_datetime(final_data['stats_dt'])


        # 重置索引
        final_data = final_data.reset_index(drop=True)

        # 记录日志
        logger.info(f"后处理后的样本数量: {len(final_data)}")

        # 更新实例的train_data
        self.data = final_data
        self.file_name = "risk_train_index_sample_data.csv"

    # 生成预测样本------------------------------------------------------------------------------------------------
    def _preprocessing_predict_data(self, predict_running_dt_end):
        """
        预处理预测数据，使用df_breed作为数据源

        参数:
        predict_running_dt_end: 预测运行结束日期
        """
        # 读取品种信息数据
        self.df_breed = pd.read_csv(config.RawData.FACT_YQ_GROSS_PROFIT_FORECAST_BREED_PATH.value)

        logger.info(f"成功读取在养鸡群数据，共 {len(self.df_breed)} 条记录")

        # 确保predict_running_dt_end为datetime类型
        if isinstance(predict_running_dt_end, str):
            predict_running_dt_end = pd.to_datetime(predict_running_dt_end)

        # 使用df_breed作为预测数据基础
        df_pred = self.df_breed.copy()

        status_map = {
            '未上市': 0,
            '已上市未完毕': 1,
            '上市完毕': 2
        }

        df_pred['status_nm'] = df_pred['status_nm'].map(status_map)

        # 添加cur_dt列，设置为predict_running_dt_end
        df_pred['cur_dt'] = predict_running_dt_end

        # 记录处理后的样本数量
        logger.info(f"预处理后的数据量: {len(df_pred)}")
        self.data = df_pred


    def _postprocessing_predict_data(self):
        """
        对预测数据进行后处理，确保只保留与训练样本相同的特征
        """
        if self.data is None or self.data.empty:
            logger.warning("没有预测数据需要后处理")
            return

        # 只保留所需特征列
        required_features = ['org_inv_dk','rearer_dk','rearer_pop_dk', 'adopt_dt', 'breeds_dk', 'rear_combin_cd', 'batch', 'cur_dt','status_nm']

        # 只保留所需特征列
        final_data = self.data[required_features].copy()

        # 确保特征类型正确
        final_data['cur_dt'] = pd.to_datetime(final_data['cur_dt'])

        # 重置索引
        final_data = final_data.reset_index(drop=True)

        # 记录日志
        logger.info(f"后处理后的样本数量: {len(final_data)}")

        # 更新实例的data
        self.data = final_data
        self.file_name = "risk_index_sample_data.csv"



    def build_train_dataset(self, train_algo_outcome_dt_end, train_algo_interval):
        """
        构建训练数据集的完整流程

        参数:
        train_algo_outcome_dt_end: 训练数据结束日期
        train_algo_interval: 训练数据时间间隔(天)

        返回:
        最终处理后的训练数据集
        """
        logger.info("-----Checking data----- ")
        self._check_data()

        logger.info("-----Preprocessing train data----- ")
        self._preprocessing_train_data(train_algo_outcome_dt_end, train_algo_interval)
        logger.info("-----Generating train data----- ")

        self._generate_samples()

        logger.info("-----Postprocessing data----- ")
        self._postprocessing_train_data()

        logger.info("-----Save as : {}".format(risk_config.algo_interim_dir / self.file_name))
        self.dump_dataset(risk_config.algo_interim_dir / self.file_name)

        return self.data
    def build_predict_dataset(self,predict_running_dt_end):
        logger.info("-----Checking data----- ")
        self._check_data()
        logger.info("-----Preprocessing predict data----- ")
        self._preprocessing_predict_data(predict_running_dt_end)
        logger.info("-----Postprocessing predict data----- ")
        self._postprocessing_predict_data()
        # logger.info("-----Save as : {}".format("/".join([prop_config.Prediction_interim_dir, self.file_name])))
        # self.dump_dataset("/".join([prop_config.Prediction_interim_dir, self.file_name]))
        logger.info("-----Done-----")
        return self.data


if __name__ == '__main__':
    pass