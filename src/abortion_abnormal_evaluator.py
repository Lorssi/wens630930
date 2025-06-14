import os
import sys
import logging
import torch
import argparse

# todo 之后工程化
from abortion_abnormal.eval.main import AbortionAbnormalPredictEval
from abortion_abnormal.module import data_pre_process_main, eval_main



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
            train_running_dt = self.task_param.get('train_running_dt')
            train_interval = self.task_param.get('train_interval')
            predict_running_dt = self.task_param.get('predict_running_dt')
            predict_interval = self.task_param.get('predict_interval')

            status_code=self.eval_generality(train_running_dt=train_running_dt,
                                             train_interval=train_interval,
                                             predict_running_dt=predict_running_dt,
                                             predict_interval=predict_interval)
            if status_code == "success":
                logger.info("测试边界时间：{}运行成功！".format(predict_running_dt))
            else:
                logger.info("测试边界时间：{}运行失败！,异常：{}".format(predict_running_dt, status_code))

            logger.info( '----------------------------------------预警测试 流程运行结束----------------------------------------')

        except Exception as e:
            # 记录错误信息
            logger.info(f"发生了一个错误: {e}", exc_info=True)
            # exit(1)

    def eval_generality(self, train_running_dt=None, train_interval=None, predict_running_dt=None, predict_interval=None):
        try:
            logger.info('----------------------------------------测试流程预处理----------------------------------------')
            # todo 数据预处理模块
            # data_pre_process = data_pre_process_main.DataPreProcessMain(running_dt=predict_running_dt, predict_interval=predict_interval,
            #                                                             logger=logger)
            # data_pre_process.check_data()
            # data_pre_process.create_eval_dataset()
           
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
            eval_main_instance = eval_main.EvalMain(eval_running_dt=predict_running_dt, eval_interval=predict_interval, logger=logger)
            eval_main_instance.eval_data()
            logger.info( '----------------------------------------测试流程运行结束----------------------------------------')
            return "success"

        except Exception as e:
            # 记录错误信息
            logger.info(f"发生了一个错误: {e}", exc_info=True)
            return e

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='This is a simple command line tool.')
    parser.add_argument('--predict_running_dt', default="2024-11-30", type=str, help='Input eval running dt end')
    parser.add_argument('--predict_interval', default=30, type=int, help='predict_interval')
    parser.add_argument('--train_running_dt_end', default="2024-05-13", type=str, help='Input train running dt end')
    args = parser.parse_args()

    predict_running_dt = args.predict_running_dt
    print(f"predict_running_dt: {predict_running_dt}")

    # & D:/data/anaconda3/envs/leb/python.exe d:/data/VSCode/wens630930/src/abortion_abnormal_evaluator.py --predict_running_dt '2024-11-30' --predict_interval 30

    # predict_running_dts = ['2024-02-29', '2024-05-31', '2024-08-31', '2024-11-30']
    # predict_running_dt = '2024-11-30'
    # feature_list = ['intro_source_num_90d', 'intro_source_is_single', 'intro_times_30d', 'intro_times_90d', 'intro_days_30d', 'intro_days_90d']
    # feature_list = ['intro_days_90d']
    
    task_param = {
        'predict_running_dt': predict_running_dt,
        'predict_interval': 30,
        # 'train_running_dt': '2023-10-01',
        # 'train_interval': 100
    }

    # 初始化评测类
    evaluator = AbortionAbnormalEvaluator(task_param=task_param)

    # 执行评测
    evaluator.eval_and_post_process()
