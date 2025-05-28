import pandas as pd
import numpy as np
from utils.logger import setup_logger
from configs.logger_config import logger_config
from tqdm import tqdm
from configs.feature_config import DataPathConfig, ColumnsConfig
logger = setup_logger(logger_config.TRAIN_LOG_FILE_PATH, logger_name="PreprocessingLogger")

class OrgDataPreprocessor:
    def __init__(self, data_path):
        self.data_path = data_path
        self.id_column = 'org_inv_dk'
        self.left_id_column = 'pigfarm_dk'

        self.org_data = self.load_data()

    def load_data(self):
        """加载数据"""
        try:
            df = pd.read_csv(self.data_path, encoding='utf-8')
            logger.info(f"成功加载组织数据: {self.data_path}")

            return df
        except FileNotFoundError:
            logger.error(f"错误: 数据文件未找到于 {self.data_path}")
            return None
        except KeyError as e:
            logger.error(f"加载数据时发生列名错误: {e}. 请检查CSV文件中的列名是否与期望的列名匹配 ({self.date_column}, {self.id_column} 等).")
            return None
        
    def get_dim_org_data(self, feature_data: pd.DataFrame):

        data = pd.merge(feature_data, self.org_data, left_on=self.left_id_column, right_on=self.id_column, how='left')

        data.to_csv(DataPathConfig.ORG_FEATURE_DATA_SAVE_PATH, index=False, encoding='utf-8')
        return data