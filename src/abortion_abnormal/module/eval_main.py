import importlib


import abortion_abnormal.eval.eval_config as config




class EvalMain:
    def __init__(self, **param):
        self.eval_running_dt = param.get('eval_running_dt')
        self.eval_interval = param.get('eval_interval')
        self.logger = param.get('logger', None)


    def eval_with_index_sample(self, eval_class_name: str, eval_main_class_name: str, eval_running_dt: str, params: dict):
        # 导包
        eval_class_module = importlib.import_module(eval_class_name)
        # 获取评估类
        eval_main_class = getattr(eval_class_module, eval_main_class_name)
        # 实例化评估类
        eval_main_class_instance = eval_main_class(logger=self.logger)
        # 构建评测数据集
        eval_main_class_instance.build_eval_set(eval_running_dt=eval_running_dt, eval_interval=self.eval_interval, **params)
        # 获取index_sample
        index_sample = eval_main_class_instance.get_eval_index_sample()
        # todo 获取预测结果
        # predict_result = predict_main_class_instance.get_predict_result()
        # 调用评估方法
        eval_main_class_instance.eval_with_index_sample(predict_data=None, save_flag=True)

        self.logger.info('索引样本数据集测试完成')


    def eval_data(self):
        self.logger.info(f'开始测试数据，时间：{self.eval_running_dt}')
        eval_class_list = config.ModulePath.eval_list.value
        for eval_module in eval_class_list:
            self.eval_with_index_sample(eval_module['eval_class_name'], eval_module['eval_main_class_name'], self.eval_running_dt, eval_module['params'])
        self.logger.info('测试数据结束')
