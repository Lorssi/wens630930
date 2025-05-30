import os
import sys
import logging
from typing import List
import pandas as pd
from pandas import DataFrame
from src.feature_alter.base_dataset import BaseDataSet
from src.transform.features import Features

# 显示所有列
pd.set_option('display.max_columns', None)
# 显示所有行
pd.set_option('display.max_rows', None)
# 设置value的显示长度为100，默认为50
pd.set_option('max_colwidth', 100)

program = os.path.basename(sys.argv[0])
logger = logging.getLogger(program)
logging.basicConfig(format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s: %(message)s',
                    datefmt='%a, %d %b %Y %H:%M:%S')
logging.root.setLevel(level=logging.INFO)


class BaseFeatureDataSet(BaseDataSet):
    def __init__(self, param: dict):
        self.entity: List[str] = []
        self.features = Features()
        self.data = DataFrame()

