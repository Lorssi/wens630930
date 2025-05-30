import os
import sys
import logging

import pandas as pd

from configs.feature_config import DataPathConfig

from feature_alter.base_dataset import BaseDataSet

# 显示所有行
pd.set_option('display.max_rows', None)
# 设置value的显示长度为100，默认为50
pd.set_option('max_colwidth', 100)

base_dir = os.path.split(os.path.realpath(__file__))[0]
program = os.path.basename(sys.argv[0])
logger = logging.getLogger(program)


class TrainIndexDataset(BaseDataSet):

    def __init__(self, **param):
        super().__init__(param)
        self.df_input = pd.read_csv(DataPathConfig.ML_DATA_PATH)
        self.data = None

        self.running_dt = param.get("running_dt", "2024-12-01")
        self.origin_feature_precompute_interval = param.get("origin_feature_precompute_interval", 400)
        
        self.end = pd.to_datetime(self.running_dt) - pd.Timedelta(days=1)
        self.start = pd.to_datetime(self.running_dt) - pd.Timedelta(days=self.origin_feature_precompute_interval)

    def _check_data(self):
        pass

    def _preprocessing_data(self):
        df_input = self.df_input[(self.df_input['date'] >= self.start) & (self.df_input['date'] <= self.end)]
        self.data = df_input

    def _postprocessing_data(self):
        self.file_name = "train_index_data.csv"

    # 生成预测样本------------------------------------------------------------------------------------------------
    def build_predict_dataset(self):
        logger.info("-----Checking Data-----")
        self._check_data()
        logger.info("-----Preprocessing Data")
        self._preprocessing_data()
        logger.info("-----Postprocessing Data-----")
        self._postprocessing_data()
        # self.dump_dataset("/".join([performance_config.Prediction_interim_dir, self.file_name]))
        logger.info("-----Done----- ")
        return self.data




if __name__ == '__main__':
    pass
