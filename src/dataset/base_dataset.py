import os
import sys
import logging

import pandas as pd

from utils import serialize, time_util

# 显示所有列
pd.set_option('display.max_columns', None)
# 显示所有行
pd.set_option('display.max_rows', None)
# 设置value的显示长度为100，默认为50
pd.set_option('max_colwidth', 100)

base_dir = os.path.split(os.path.realpath(__file__))[0]
program = os.path.basename(sys.argv[0])
logger = logging.getLogger(program)

logging.basicConfig(format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s: %(message)s',
                    datefmt='%a, %d %b %Y %H:%M:%S')
logging.root.setLevel(level=logging.INFO)


def _split_sequence(line, delimiter="#"):
    return line.strip().split(delimiter) if isinstance(line, str) else []


class BaseDataSet:

    def __init__(self, param: dict):
        self.dump_path = param.get("dump_path", "./")
        self.file_type = param.get("file_type", "csv")
        self.data = None

    def dump_dataset(self, dataset_file):
        serialize.dataframe_dump(self.data, dataset_file, self.file_type)

    def load_dataset(self, dataset_file):
        self.data = serialize.dataframe_read(dataset_file, self.file_type)

    @time_util.func_time_cost
    def build_dataset(self, **param):
        # data info
        # join data
        return 0
