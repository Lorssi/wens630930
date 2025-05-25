import os
import sys
import logging
import torch

# todo 之后工程化
from feature.gen_feature import FeatureGenerator
from main_train import main_train
from main_predict import main_predict
from abortion_abnormal.eval.main import AbortionAbnormalAllOnsetEval



base_dir = os.path.split(os.path.realpath(__file__))[0]
program = os.path.basename(sys.argv[0])
logger = logging.getLogger(program)


class AbortionAbnormalEvaluator:
    def __init__(self, **param):
        self.task_param = param.get('task_param')

    def eval_and_post_process(self):
        logger.info(f"abortion_abnormal initialized with parameters: {self.task_param}")
        try:
            logger.info('----------------------------------------开始预警 执行测试流程----------------------------------------')
            logger.info(torch.cuda.is_available())
            # 参数初始化
            predict_running_dt_end = self.task_param.get('predict_running_dt_end')
            predict_interval = self.task_param.get('predict_interval')
            train_running_dt_end = self.task_param.get('train_running_dt_end')
            train_interval = self.task_param.get('train_interval')

            status_code=self.eval_generality(train_running_dt_end=train_running_dt_end,
                                             train_interval=train_interval,
                                             predict_running_dt_end=predict_running_dt_end,
                                             predict_interval=predict_interval)
            if status_code == "success":
                logger.info("测试边界时间：{}运行成功！".format(predict_running_dt_end))
            else:
                logger.info("测试边界时间：{}运行失败！,异常：{}".format(predict_running_dt_end, status_code))

            logger.info( '----------------------------------------预警测试 流程运行结束----------------------------------------')

        except Exception as e:
            # 记录错误信息
            logger.info(f"发生了一个错误: {e}", exc_info=True)
            # exit(1)

    def eval_generality(self, train_running_dt_end=None, train_interval=None, predict_running_dt_end=None, predict_interval=None):
        try:
            logger.info('----------------------------------------测试流程预处理----------------------------------------')
            # todo 数据预处理模块
            # data_pre_process = data_pre_process_main.DataPreProcessMain(running_dt=predict_running_dt_end,
            #                                                             logger=logger)
            # data_pre_process.check_data()
           
            logger.info( '----------------------------------------测试特征预计算----------------------------------------')
            # todo 特征预计算模块
            # feature_generator = FeatureGenerator(running_dt=predict_running_dt_end,
            #                                      interval_days=train_interval)
            # feature_df = feature_generator.generate_features()

            logger.info('----------------------------------------测试模型训练----------------------------------------')
            # todo 模型训练模块
            # main_train(train_running_dt=train_running_dt_end)

            logger.info('----------------------------------------测试数据预测----------------------------------------')
            # todo 数据预测
            # main_predict(predict_running_dt=predict_running_dt_end, predict_interval=predict_interval)

            logger.info('----------------------------------------测试模型评估----------------------------------------')
            # todo 模型评估模块
            abortion_abnormal_eval = AbortionAbnormalAllOnsetEval(logger=logger)
            abortion_abnormal_eval.build_eval_set(eval_running_dt_end=predict_running_dt_end, eval_interval=predict_interval)
            abortion_abnormal_eval.eval_with_index_sample()
            logger.info( '----------------------------------------测试流程运行结束----------------------------------------')
            return "success"

        except Exception as e:
            # 记录错误信息
            logger.info(f"发生了一个错误: {e}", exc_info=True)
            return e

if __name__ == "__main__":
    predict_running_dt_end_list = ["2023-06-13", "2023-07-29", "2023-09-13", "2023-10-29", "2023-12-14", "2024-01-29", "2024-03-15", "2024-04-24"]
    train_running_dt_end_list = ["2023-5-15", "2023-6-30", "2023-8-15", "2023-9-30", "2023-11-15", "2023-12-31", "2024-2-15", "2024-3-26"]

    # Create task parameters dictionary
    task_param = {
        'predict_running_dt_end': '2023-06-13',
        'predict_interval': 28,
        'train_running_dt_end': '2023-05-15',
        'train_interval': 100
    }

    # task_param = {
    #     'predict_running_dt_end': '2025-03-01',
    #     'predict_interval': 90,
    #     'train_running_dt_end': '2024-10-01',
    #     'train_interval': 100
    # }

    # Initialize the evaluator
    evaluator = AbortionAbnormalEvaluator(task_param=task_param)

    # Run the evaluation process
    evaluator.eval_and_post_process()

    
