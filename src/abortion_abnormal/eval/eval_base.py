import os, sys, logging
from dataclasses import dataclass
from dataclasses_json import dataclass_json
import pandas as pd


base_dir = os.path.split(os.path.realpath(__file__))[0]
program = os.path.basename(sys.argv[0])
logger = logging.getLogger(program)
logging.basicConfig(format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s: %(message)s',
                    datefmt='%a, %d %b %Y %H:%M:%S')
logging.root.setLevel(level=logging.INFO)


@dataclass_json
@dataclass
class EvalBaseMixin(object):

    def build_eval_set(self, eval_running_dt_end, outcome_offset: int, outcome_window_len: int, eval_interval: int,
                       param=None):
        pass

    def get_eval_index_sample(self):
        pass

    def eval_with_dt(self, predict_result: pd.DataFrame):
        pass

    def eval_with_index_sample(self, predict_result: pd.DataFrame):
        pass


if __name__ == '__main__':
    pass