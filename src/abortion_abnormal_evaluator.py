import os
import sys
import logging
import torch
import argparse

# todo 之后工程化
from abortion_abnormal.eval.main import AbortionAbnormalEval1
from abortion_abnormal.module import data_pre_process_main



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
            logger.info('是否存在显卡:', torch.cuda.is_available())
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
            # todo 模型评估模块
            abortion_abnormal_eval = AbortionAbnormalEval1(logger=logger)
            abortion_abnormal_eval.build_eval_set(eval_running_dt=predict_running_dt, eval_interval=predict_interval)
            abortion_abnormal_eval.eval_with_index_sample()
            logger.info( '----------------------------------------测试流程运行结束----------------------------------------')
            return "success"

        except Exception as e:
            # 记录错误信息
            logger.info(f"发生了一个错误: {e}", exc_info=True)
            return e

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='This is a simple command line tool.')
    parser.add_argument('--predict_running_dt', default="2024-06-13", type=str, help='Input eval running dt end')
    parser.add_argument('--predict_interval', default=28, type=int, help='predict_interval')
    parser.add_argument('--train_running_dt_end', default="2024-05-13", type=str, help='Input train running dt end')
    args = parser.parse_args()

    # & D:/data/anaconda3/envs/leb/python.exe d:/data/VSCode/wens630930/src/abortion_abnormal_evaluator.py --predict_running_dt '2024-11-30' --predict_interval 30

    # task_param = {
    #     'predict_running_dt': args.predict_running_dt,
    #     'predict_interval': args.predict_interval,
    #     'train_running_dt_end': args.train_running_dt_end,
    # }

    # 统一 todo 将default改为流产率预测类
    # task_param = {
    #     'predict_running_dt': '2024-06-13',
    #     'predict_interval': 21,
    #     'train_running_dt': '2024-05-15',
    #     # 'train_interval': 100
    # }

    # task_param = {
    #     'predict_running_dt': '2025-03-01',
    #     'predict_interval': 90,
    #     'train_running_dt': '2024-10-01',
    #     'train_interval': 100
    # }

    # predict_running_dts = ['2024-02-29', '2024-05-31', '2024-08-31', '2024-11-30']
    
    # for predict_running_dt in predict_running_dts:
    task_param = {
        'predict_running_dt': '2024-05-31',
        'predict_interval': 30,
        # 'train_running_dt': '2024-10-01',
        # 'train_interval': 100
    }

    # 初始化评测类
    evaluator = AbortionAbnormalEvaluator(task_param=task_param)

    # 执行评测
    evaluator.eval_and_post_process()
