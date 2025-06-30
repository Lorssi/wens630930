import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, classification_report
from sklearn.exceptions import UndefinedMetricWarning
from tqdm import tqdm
import warnings
import logging
import os

import sys
from pathlib import Path
# 获取项目根目录（假设是 src 的父目录）
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_root))

program = os.path.basename(sys.argv[0])
eval_logger = logging.getLogger(program)
logging.basicConfig(format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s: %(message)s',
                    datefmt='%a, %d %b %Y %H:%M:%S')
logging.root.setLevel(level=logging.INFO)


class AbortionAbnormalPredictEvalData():

    def __init__(self, predict_result, sample_ground_truth, eval_running_dt_start, eval_running_dt_end, logger=None):
        """
        初始化评估数据
        :param predict_data: 预测数据 DataFrame
        :param eval_running_dt_start: 评估开始日期
        :param eval_running_dt_end: 评估结束日期
        :param logger: 日志记录器
        """
        self.ground_truth_sample_num = len(sample_ground_truth)
        self.predict_sample_num = len(predict_result)
        self.recognition = self.ground_truth_sample_num / self.predict_sample_num if self.predict_sample_num > 0 else 0

        self.predict_data = sample_ground_truth.merge(
            predict_result,
            on=['stats_dt', 'pigfarm_dk'],
            how='left'
        )

        self.eval_running_dt_start = pd.to_datetime(eval_running_dt_start)
        self.eval_running_dt_start_string = self.eval_running_dt_start.strftime('%Y-%m-%d')
        self.eval_running_dt_end = pd.to_datetime(eval_running_dt_end)
        self.eval_running_dt_end_string = self.eval_running_dt_end.strftime('%Y-%m-%d')
        if logger is None:
            self.logger = eval_logger
        else:
            self.logger = logger
        self.feiwen_dict = None
        self.truth_column = 'abort_{}'
        self.predict_column = 'abort_{}_decision'
        self.probability_column = 'abort_{}_pred'


    def exclude_feiwen_data(self, sample, period):
        """
        排除与非洲猪瘟时间有交集的样本
        :param sample: 输入样本数据
        :param period: 预测周期，'1_7', '8_14', '15_21'
        :return: 过滤后的样本数据
        """
        # 根据period确定预测范围
        if period == '1_7':
            start, end = 1, 7
        elif period == '8_14':
            start, end = 8, 14
        elif period == '15_21':
            start, end = 15, 21

        # 缓存，加快速度
        if self.feiwen_dict is None:
            # 获取非瘟数据
            feiwen_data = pd.read_csv("data/raw_data/ADS_PIG_FARM_AND_REARER_ONSET.csv", encoding='utf-8')
            # 只保留需要的列
            feiwen_data = feiwen_data[['org_inv_dk', 'start_dt', 'end_dt']]
            # 转换日期格式
            feiwen_data['start_dt'] = pd.to_datetime(feiwen_data['start_dt'])
            feiwen_data['end_dt'] = pd.to_datetime(feiwen_data['end_dt'])
            # 确保ID为字符串类型
            feiwen_data['org_inv_dk'] = feiwen_data['org_inv_dk'].astype(str)

            # 确保ID为字符串类型
            sample = sample.copy()
            sample['pigfarm_dk'] = sample['pigfarm_dk'].astype(str)

            # 按猪场ID对非瘟数据进行分组
            feiwen_dict = {}
            for farm_id, group in feiwen_data.groupby('org_inv_dk'):
                feiwen_dict[farm_id] = group[['start_dt', 'end_dt']].values.tolist()
            self.feiwen_dict = feiwen_dict

        feiwen_dict = self.feiwen_dict

        # 计算样本的预测开始和结束日期
        sample['predict_start'] = sample['stats_dt'] + pd.Timedelta(days=start)
        sample['predict_end'] = sample['stats_dt'] + pd.Timedelta(days=end)

        # 创建一个标记是否为非瘟样本的列
        sample['is_feiwen'] = False

        # 检查每个样本是否与非瘟时间有交集
        for idx, row in sample.iterrows():
            farm_id = row['pigfarm_dk']

            # 如果该猪场有非瘟记录，则检查是否有时间交集
            if farm_id in feiwen_dict:
                predict_start = row['predict_start']
                predict_end = row['predict_end']

                # 检查是否与任何一个非瘟时间段有交集
                for feiwen_start, feiwen_end in feiwen_dict[farm_id]:
                    feiwen_start = pd.to_datetime(feiwen_start)
                    feiwen_end = pd.to_datetime(feiwen_end)
                    # 有交集的条件: 非瘟开始日期 <= 预测结束日期 且 预测开始日期 <= 非瘟结束日期
                    if feiwen_start <= predict_end and predict_start <= feiwen_end:
                        sample.at[idx, 'is_feiwen'] = True
                        break

        # 过滤掉与非瘟时间有交集的样本
        filtered_sample = sample[~sample['is_feiwen']]

        # 删除临时列
        filtered_sample = filtered_sample.drop(['predict_start', 'predict_end', 'is_feiwen'], axis=1)

        return filtered_sample


    def get_special_sample(self, data, period, prev_status, next_status):
        """
        获取特殊样本
        :param period: 预测周期，'1_7', '8_14', '15_21'
        :param prev_status: 前一天的状态，0或1
        :param next_status: 后一天的状态，0或1
        :return: 特殊样本 DataFrame
        """
        # 初始化特殊样本列表
        special_samples = []

        # 按照猪场分组
        grouped = data.groupby('pigfarm_dk')

        if (prev_status == 0 and next_status == 1) or (prev_status == 1 and next_status == 0): # 获取从正常到异常或从异常到正常的样本
            for pigfarm, group_data in grouped:
                # 按时间排序
                sorted_data = group_data.sort_values('stats_dt')

                # 找出时间t时abort_period为0(1)，时间t+1时abort_period为1(0)的时间点
                for i in range(len(sorted_data) - 1):
                    if sorted_data.iloc[i][self.truth_column.format(period)] == prev_status and sorted_data.iloc[i+1][self.truth_column.format(period)] == next_status:
                        # 获取时间点t
                        t = sorted_data.iloc[i]['stats_dt']

                        # 找出t-1到t+2的所有样本
                        farm_data = sorted_data.copy()
                        t_minus_1 = t - pd.Timedelta(days=1)
                        t_plus_1 = t + pd.Timedelta(days=1)
                        t_plus_2 = t + pd.Timedelta(days=2)

                        # 筛选时间范围内的样本
                        time_range_samples = farm_data[(farm_data['stats_dt'] >= t_minus_1) &
                                                        (farm_data['stats_dt'] <= t_plus_2)]

                        # 判断是否符合连续性，即两天的样本是相同的
                        t_minus_1_sample = time_range_samples[time_range_samples['stats_dt'] == t_minus_1]
                        t_sample = time_range_samples[time_range_samples['stats_dt'] == t]
                        t_plus_1_sample = time_range_samples[time_range_samples['stats_dt'] == t_plus_1]
                        t_plus_2_sample = time_range_samples[time_range_samples['stats_dt'] == t_plus_2]

                        # 检查是否存在这两天的样本
                        if not t_minus_1_sample.empty and not t_sample.empty:
                            # 获取 abort_{period} 值
                            t_minus_1_abort = t_minus_1_sample.iloc[0][self.truth_column.format(period)]
                            t_abort = t_sample.iloc[0][self.truth_column.format(period)]

                            # 如果两天的 abort_{period} 值不同，删除 t_minus_1 的数据
                            if t_minus_1_abort != t_abort:
                                time_range_samples = time_range_samples[time_range_samples['stats_dt'] != t_minus_1]

                        # 如果 t+1 和 t+2 的样本存在，检查它们的 abort_{period} 值
                        if not t_plus_1_sample.empty and not t_plus_2_sample.empty:
                            t_plus_1_abort = t_plus_1_sample.iloc[0][self.truth_column.format(period)]
                            t_plus_2_abort = t_plus_2_sample.iloc[0][self.truth_column.format(period)]
                            # 如果两天的 abort_{period} 值不同，删除 t+2 的数据
                            if t_plus_1_abort != t_plus_2_abort:
                                time_range_samples = time_range_samples[time_range_samples['stats_dt'] != t_plus_2]

                        # 添加到特殊样本列表
                        special_samples.append(time_range_samples)

        elif prev_status == 1 and next_status == 1: # 获取从异常到异常的样本
            special_samples = data[data[self.truth_column.format(period)] == 1].copy()
            return special_samples

        elif prev_status == 1 and next_status == 2: # 获取被异常流产率影响的从正常到异常样本
            for pigfarm, group_data in grouped:
                # 按时间排序
                sorted_data = group_data.sort_values('stats_dt')

                # 找出时间t时abort_period为0(1)，时间t+1时abort_period为1(0)的时间点
                for i in range(len(sorted_data) - 1):
                    if sorted_data.iloc[i][self.truth_column.format(period)] == 0 and sorted_data.iloc[i+1][self.truth_column.format(period)] == 1:
                        # 获取时间点t
                        t = sorted_data.iloc[i]['stats_dt']

                        # 找出t-1到t+2的所有样本
                        farm_data = sorted_data.copy()
                        t_minus_1 = t - pd.Timedelta(days=1)
                        t_plus_1 = t + pd.Timedelta(days=1)
                        t_plus_2 = t + pd.Timedelta(days=2)

                        # 筛选时间范围内的样本
                        time_range_samples = farm_data[(farm_data['stats_dt'] >= t_minus_1) &
                                                        (farm_data['stats_dt'] <= t_plus_2)]

                        # 判断是否符合连续性，即两天的样本是相同的
                        t_minus_1_sample = time_range_samples[time_range_samples['stats_dt'] == t_minus_1]
                        t_sample = time_range_samples[time_range_samples['stats_dt'] == t]
                        t_plus_1_sample = time_range_samples[time_range_samples['stats_dt'] == t_plus_1]
                        t_plus_2_sample = time_range_samples[time_range_samples['stats_dt'] == t_plus_2]

                        # 检查是否存在这两天的样本
                        if not t_minus_1_sample.empty and not t_sample.empty:
                            # 获取 abort_{period} 值
                            t_minus_1_abort = t_minus_1_sample.iloc[0][self.truth_column.format(period)]
                            t_abort = t_sample.iloc[0][self.truth_column.format(period)]

                            # 如果两天的 abort_{period} 值不同，删除 t_minus_1 的数据
                            if t_minus_1_abort != t_abort:
                                time_range_samples = time_range_samples[time_range_samples['stats_dt'] != t_minus_1]

                        # 如果 t+1 和 t+2 的样本存在，检查它们的 abort_{period} 值
                        if not t_plus_1_sample.empty and not t_plus_2_sample.empty:
                            t_plus_1_abort = t_plus_1_sample.iloc[0][self.truth_column.format(period)]
                            t_plus_2_abort = t_plus_2_sample.iloc[0][self.truth_column.format(period)]
                            # 如果两天的 abort_{period} 值不同，删除 t+2 的数据
                            if t_plus_1_abort != t_plus_2_abort:
                                time_range_samples = time_range_samples[time_range_samples['stats_dt'] != t_plus_2]

                        # 检查所有样本的 single_influence_{period} 是否都为 1
                        influence_column = f'single_influence_{period}'
                        if influence_column in time_range_samples.columns:
                            if time_range_samples[influence_column].all():  # 确保所有值都为 True/1
                                # 添加到特殊样本列表
                                special_samples.append(time_range_samples)
                        else:
                            # 如果列不存在，记录警告
                            if self.logger:
                                self.logger.warning(f"列 {influence_column} 不存在，无法检查影响因素")

        elif prev_status == 0 and next_status == 2: # 获取猪业一部从正常的异常的样本
            for pigfarm, group_data in grouped:
                # 按时间排序
                sorted_data = group_data.sort_values('stats_dt')

                # 找出时间t时abort_period为0(1)，时间t+1时abort_period为1(0)的时间点
                for i in range(len(sorted_data) - 1):
                    if sorted_data.iloc[i][self.truth_column.format(period)] == prev_status and sorted_data.iloc[i+1][self.truth_column.format(period)] == next_status:
                        # 获取时间点t
                        t = sorted_data.iloc[i]['stats_dt']

                        # 找出t-1到t+2的所有样本
                        farm_data = sorted_data.copy()
                        t_minus_1 = t - pd.Timedelta(days=1)
                        t_plus_1 = t + pd.Timedelta(days=1)
                        t_plus_2 = t + pd.Timedelta(days=2)

                        # 筛选时间范围内的样本
                        time_range_samples = farm_data[(farm_data['stats_dt'] >= t_minus_1) &
                                                        (farm_data['stats_dt'] <= t_plus_2)]

                        # 判断是否符合连续性，即两天的样本是相同的
                        t_minus_1_sample = time_range_samples[time_range_samples['stats_dt'] == t_minus_1]
                        t_sample = time_range_samples[time_range_samples['stats_dt'] == t]
                        t_plus_1_sample = time_range_samples[time_range_samples['stats_dt'] == t_plus_1]
                        t_plus_2_sample = time_range_samples[time_range_samples['stats_dt'] == t_plus_2]

                        # 检查是否存在这两天的样本
                        if not t_minus_1_sample.empty and not t_sample.empty:
                            # 获取 abort_{period} 值
                            t_minus_1_abort = t_minus_1_sample.iloc[0][self.truth_column.format(period)]
                            t_abort = t_sample.iloc[0][self.truth_column.format(period)]

                            # 如果两天的 abort_{period} 值不同，删除 t_minus_1 的数据
                            if t_minus_1_abort != t_abort:
                                time_range_samples = time_range_samples[time_range_samples['stats_dt'] != t_minus_1]

                        # 如果 t+1 和 t+2 的样本存在，检查它们的 abort_{period} 值
                        if not t_plus_1_sample.empty and not t_plus_2_sample.empty:
                            t_plus_1_abort = t_plus_1_sample.iloc[0][self.truth_column.format(period)]
                            t_plus_2_abort = t_plus_2_sample.iloc[0][self.truth_column.format(period)]
                            # 如果两天的 abort_{period} 值不同，删除 t+2 的数据
                            if t_plus_1_abort != t_plus_2_abort:
                                time_range_samples = time_range_samples[time_range_samples['stats_dt'] != t_plus_2]

                        # 添加到特殊样本列表
                        special_samples.append(time_range_samples)

        else:
            raise ValueError("prev_status 和 next_status 的组合不合法，仅支持0或1")

        # 合并所有特殊样本
        if len(special_samples) == 0:
            self.logger.warning(f"没有找到满足条件的特殊样本，abortion_period: {period}, prev_status: {prev_status}, next_status: {next_status}")
            return pd.DataFrame()

        return pd.concat(special_samples)


    def calculate_special_recall(self, special_samples, period):
        """
        计算特殊的召回率指标
        按时间段粒度计算：先将样本按照时间是否连续分为多个时间段，然后计算每个时间段的recall，再计算每个猪场的平均recall，最后计算所有猪场的平均recall
        """
        # 检查是否有样本
        if len(special_samples) == 0:
            if self.logger:
                self.logger.warning("没有找到符合条件的样本用于计算特殊召回率")
            return 0

        # 保存每个猪场的recall值
        farm_recalls = {}

        # 按照猪场分组
        for pigfarm, group_data in special_samples.groupby('pigfarm_dk'):
            # 按时间排序
            sorted_data = group_data.sort_values('stats_dt')

            # 初始化时间段列表和时间段的recall列表
            time_periods = []
            current_period = []
            period_recalls = []

            # 按时间连续性分组
            if len(sorted_data) > 0:
                # 初始化第一个时间段
                current_date = sorted_data.iloc[0]['stats_dt']
                current_period = [sorted_data.iloc[0]]

                # 遍历后续日期，根据连续性分组
                for i in range(1, len(sorted_data)):
                    next_date = sorted_data.iloc[i]['stats_dt']

                    # 如果日期连续（相差1天）则添加到当前时间段
                    if (next_date - current_date).days == 1:
                        current_period.append(sorted_data.iloc[i])
                    else:
                        # 如果不连续，保存当前时间段并开始新的时间段
                        if current_period:
                            time_periods.append(pd.DataFrame(current_period))
                        current_period = [sorted_data.iloc[i]]

                    current_date = next_date

                # 添加最后一个时间段
                if current_period:
                    time_periods.append(pd.DataFrame(current_period))

            # 计算每个时间段的recall
            for period_data in time_periods:
                # 获取当前时间段的预测周期，假设所有样本使用相同的预测周期

                # 针对每个预测周期计算recall
                y_true = period_data[self.truth_column.format(period)].values
                y_pred = period_data[self.predict_column.format(period)].values

                # 只有当实际有阳性样本时才能计算recall
                if sum(y_true) > 0:
                    period_recall = recall_score(y_true, y_pred, zero_division=0)
                    period_recalls.append(period_recall)

            # 计算该猪场的平均recall
            if period_recalls:
                farm_recalls[pigfarm] = sum(period_recalls) / len(period_recalls)

        # 计算所有猪场的平均recall
        if farm_recalls:
            special_recall = sum(farm_recalls.values()) / len(farm_recalls)
        else:
            special_recall = 0

        return special_recall


    def special_sample_1_eval_one_periods_metric(self, period, prev_status, next_status):
        """
        特殊样本1：计算前一天为prev_status(01)，后一天为next_status(10)的评估指标
        :param period: 预测周期，'1_7', '8_14', '15_21'
        :param prev_status: 前一天的状态，0或1
        :param next_status: 后一天的状态，0或1
        """
        if prev_status == 0 and next_status == 1:
            sample_type = 'normal-2_to_abnormal-2'
        elif prev_status == 1 and next_status == 0:
            sample_type = 'abnormal-2_to_normal-2'
        elif prev_status == 1 and next_status == 2: # delete 这一行只是写着方便，与实际意义无关
            sample_type = 'normal-2_to_abnormal-2_single_influence'
        # 1. 筛选特殊样本
        # 获取预测数据
        data = self.predict_data.copy()

        special_samples = self.get_special_sample(data, period, prev_status, next_status)
        # if prev_status == 0 and next_status == 1:
        #     special_samples.to_csv(f'special_samples_{period}.csv', index=False)

        # 合并所有特殊样本
        if len(special_samples) > 0:
            # 2. 去重
            special_samples = special_samples.drop_duplicates()
            # 计算原始样本数量
            total_samples_num = len(special_samples)
            # 3. 剔除非瘟数据
            special_samples = self.exclude_feiwen_data(special_samples, period)
            # 计算剔除后的样本数量
            filtered_samples_num = len(special_samples)
        else:
            self.logger.warning(f"没有找到满足条件的特殊样本，abortion_period: {period}")
            return

        if prev_status == 0 and next_status == 1:
            special_samples.to_csv(f'special_samples_exclude_feiwen_{period}.csv', index=False, encoding='utf-8')

        # 3. 计算评估指标
        y_true = special_samples[self.truth_column.format(period)].values
        y_pred = special_samples[self.predict_column.format(period)].values
        y_prob = special_samples[self.probability_column.format(period)].values

        # 计算指标
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)

        # 计算AUC
        auc = 0
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
                auc = roc_auc_score(y_true, y_prob)
        except:
            # 处理AUC计算异常情况（例如只有一个类别）
            if self.logger:
                self.logger.warning("无法计算AUC，可能是样本中只存在一个类别")

        # 4. 计算special_recall（时间段粒度）
        special_recall = self.calculate_special_recall(special_samples, period)

        # 保存结果
        result_row = {
            'stats_dt': f"{self.eval_running_dt_start_string} ~ {self.eval_running_dt_end_string}",
            'sample_type': sample_type,
            'total_sample_num': total_samples_num,
            'remain_sample_num': filtered_samples_num,
            'eval_period': period,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'auc': auc,
            'special_recall': special_recall,
        }
        self.result = pd.concat([self.result, pd.DataFrame([result_row])], ignore_index=True)


    def special_sample_2_eval_one_periods_metric(self, period):
        """
        特殊样本2：计算真实结果为1
        :param period: 预测周期，'1_7', '8_14', '15_21'
        """
        sample_type = 'abnormal_to_abnormal'

        # 1. 筛选特殊样本 - 所有真实值为1的样本
        data = self.predict_data.copy()
        special_samples = self.get_special_sample(data, period, 1, 1)

        # 计算原始样本数量
        total_samples_num = len(special_samples)

        if total_samples_num == 0:
            if self.logger:
                self.logger.warning(f"没有找到真实标签为1的样本，abortion_period: {period}")
            return

        # 2. 去重
        special_samples = special_samples.drop_duplicates()

        # 3. 剔除非瘟数据
        special_samples = self.exclude_feiwen_data(special_samples, period)

        # 计算剔除后的样本数量
        filtered_samples_num = len(special_samples)

        # 4. 计算评估指标
        y_true = special_samples[self.truth_column.format(period)].values
        y_pred = special_samples[self.predict_column.format(period)].values
        y_prob = special_samples[self.probability_column.format(period)].values

        # 计算指标
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)

        # 计算AUC
        auc = 0
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
                auc = roc_auc_score(y_true, y_prob)
        except:
            # 处理AUC计算异常情况（例如只有一个类别）
            if self.logger:
                self.logger.warning("无法计算AUC，可能是样本中只存在一个类别")

        # 5. 计算special_recall（时间段粒度）
        special_recall = self.calculate_special_recall(special_samples, period)

        # 保存结果
        result_row = {
            'stats_dt': f"{self.eval_running_dt_start_string} ~ {self.eval_running_dt_end_string}",
            'sample_type': sample_type,
            'total_sample_num': total_samples_num,
            'remain_sample_num': filtered_samples_num,
            'eval_period': period,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'auc': auc,
            'special_recall': special_recall
        }
        self.result = pd.concat([self.result, pd.DataFrame([result_row])], ignore_index=True)


    def overall_eval_one_periods_metric(self, period, exclude_feiwen=True, hierarchical_data=None, level=None, name=None):
        """
        计算整体指标
        :param abortion_period: 预测周期，'1_7', '8_14', '15_21'
        :param exclude_feiwen: 是否排除非洲猪瘟数据
        """
        # 获取数据
        if hierarchical_data is not None:
            data = hierarchical_data.copy()
        else:
            data = self.predict_data.copy()

        # 计算原始样本数量
        total_samples_num = len(data)

        # 如果需要排除非瘟数据
        if exclude_feiwen:
            # 剔除非瘟数据
            data = self.exclude_feiwen_data(data, period)
            if hierarchical_data is None:
                data.to_csv(f'exclude_feiwen_data_{period}.csv', index=False, encoding='utf-8')


        # 计算剔除后的样本数量
        filtered_samples_num = len(data)

        # 获取标签和预测值
        y_true = data[self.truth_column.format(period)].values
        y_pred = data[self.predict_column.format(period)].values
        y_prob = data[self.probability_column.format(period)].values

        # 计算基础指标
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)

        # 计算AUC
        auc = 0
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
                auc = roc_auc_score(y_true, y_prob)
        except:
            # 处理AUC计算异常情况（例如只有一个类别）
            if self.logger:
                self.logger.warning(f"无法计算{period}的AUC，可能是样本中只存在一个类别")

        # 保存结果
        result_row = {
            'stats_dt': f"{self.eval_running_dt_start_string} ~ {self.eval_running_dt_end_string}",
            f'l{level}_name': name,
            'total_sample_num': total_samples_num,
            'remain_sample_num': filtered_samples_num,
            'eval_period': period,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'auc': auc,
            'recognition': self.recognition
        }

        result_df = pd.DataFrame([result_row])
        if not exclude_feiwen:
            result_df.drop(columns=['remain_sample_num'], inplace=True)
        if hierarchical_data is None:
            result_df.drop(columns=[f'l{level}_name'], inplace=True)

        self.result = pd.concat([self.result, result_df], ignore_index=True)


    def eval_with_organizational_hierarchy(self):
        """计算不同组织层级的评估指标"""

        # 1. L2级别组织评估
        self.result = pd.DataFrame()
        l2_groups = self.predict_data.groupby('l2_org_inv_nm')

        for l2_name, l2_data in tqdm(l2_groups, desc='L2级别组织评估'):
            # 对每个时间段进行评估
            for abortion_period in ['1_7', '8_14', '15_21']:
                # 使用this l2组的数据进行评估
                self.overall_eval_one_periods_metric(abortion_period, exclude_feiwen=True, hierarchical_data=l2_data, level=2, name=l2_name)

        l2_results = self.result.copy()

        # 2. L3级别组织评估
        self.result = pd.DataFrame()
        l3_groups = self.predict_data.groupby('l3_org_inv_nm')

        for l3_name, l3_data in tqdm(l3_groups, desc='L3级别组织评估'):
            for abortion_period in ['1_7', '8_14', '15_21']:
                self.overall_eval_one_periods_metric(abortion_period, exclude_feiwen=True, hierarchical_data=l3_data, level=3, name=l3_name)

        l3_results = self.result.copy()

        # 3. L4级别组织评估
        self.result = pd.DataFrame()
        l4_groups = self.predict_data.groupby('l4_org_inv_nm')

        for l4_name, l4_data in tqdm(l4_groups, desc='L4级别组织评估'):
            # 跳过缺失的组织名称
            if pd.isna(l4_name):
                continue
            for abortion_period in ['1_7', '8_14', '15_21']:
                self.overall_eval_one_periods_metric(abortion_period, exclude_feiwen=True, hierarchical_data=l4_data, level=4, name=l4_name)

        l4_results = self.result.copy()

        return l2_results, l3_results, l4_results


    # 计算整体指标
    def eval_all_samples(self):
        self.result = pd.DataFrame() # 清空上一次的结果
        for abortion_period in tqdm(['1_7', '8_14', '15_21'], desc='计算整体指标'):
            self.overall_eval_one_periods_metric(abortion_period, exclude_feiwen=False)

        return self.result

    # 计算剔除非瘟整体指标
    def eval_all_samples_exclude_feiwen(self):
        self.result = pd.DataFrame() # 清空上一次的结果
        for abortion_period in tqdm(['1_7', '8_14', '15_21'], desc='计算剔除非瘟整体指标'):
            self.overall_eval_one_periods_metric(abortion_period)

        return self.result

    # 计算特殊样本指标
    def eval_special_samples(self, l2_name=None):
        tmp_data = self.predict_data.copy() # 备份原始数据
        if l2_name is not None:
            # 如果指定了l2_name，则只计算该组织下的特殊样本
            self.predict_data = self.predict_data[self.predict_data['l2_org_inv_nm'].isin(l2_name)]

        ## 计算abnormal-2_to_normal-2和normal-2_to_abnormal-2指标
        self.result = pd.DataFrame() # 清空上一次的结果
        for prev_status, next_status in [(0, 1), (1, 0)]:
            for abortion_period in tqdm(['1_7', '8_14', '15_21'], desc=f'计算{prev_status}到{next_status}特殊样本1指标'):
                # 评估从低到高的指标
                self.special_sample_1_eval_one_periods_metric(abortion_period, prev_status, next_status)

        ## 计算abnormal_to_abnormal指标
        for abortion_period in tqdm(['1_7', '8_14', '15_21'], desc='计算特殊样本2指标'):
            self.special_sample_2_eval_one_periods_metric(abortion_period)

        ## 计算排除异常流产率的数据的normal_to_abnormal指标
        for abortion_period in tqdm(['1_7', '8_14', '15_21'], desc='计算normal_to_abnormal排除异常流产率指标'):
            self.special_sample_1_eval_one_periods_metric(abortion_period, 1, 2)

        self.predict_data = tmp_data # 恢复原始数据
        return self.result

    # 计算分级组织指标
    def eval_organizational_hierarchy(self):
        l2_results, l3_results, l4_results = self.eval_with_organizational_hierarchy()
        return l2_results, l3_results, l4_results

    # 计算每个猪场的流产持续时间数量和持续时间平均值
    def calculate_abortion_duration(self):
        """
        计算每个猪场的流产持续时间数量和持续时间平均值
        :return: DataFrame 包含每个猪场的流产持续时间数量和平均持续时间
        """
        ground_truth = self.predict_data.copy()

        # 创建结果字典，用于存储每个猪场的数据
        farm_data = {}

        # 按猪场分组
        for pigfarm, group_data in ground_truth.groupby('pigfarm_dk'):
            farm_data[pigfarm] = {
                'pigfarm_dk': pigfarm,
                '1_7_num_abortion_periods': 0,
                '1_7_avg_duration_days': 0,
                '8_14_num_abortion_periods': 0,
                '8_14_avg_duration_days': 0,
                '15_21_num_abortion_periods': 0,
                '15_21_avg_duration_days': 0
            }

            # 处理每种流产周期类型
            for period in ['1_7', '8_14', '15_21']:
                # 筛选出abort_{period}为1的样本
                if self.truth_column.format(period) not in group_data.columns:
                    continue

                abortion_samples = group_data[group_data[self.truth_column.format(period)] == 1].sort_values('stats_dt')

                if len(abortion_samples) == 0:
                    continue

                # 识别连续的流产周期
                abortion_periods = []
                current_period = [abortion_samples.iloc[0]['stats_dt']]

                # 检查日期连续性
                for i in range(1, len(abortion_samples)):
                    curr_date = abortion_samples.iloc[i]['stats_dt']
                    prev_date = abortion_samples.iloc[i-1]['stats_dt']

                    # 判断是否连续（日期相差1天）
                    if (curr_date - prev_date).days == 1:
                        # 连续，添加到当前周期
                        current_period.append(curr_date)
                    else:
                        # 不连续，结束当前周期并开始新周期
                        abortion_periods.append(current_period)
                        current_period = [curr_date]

                # 添加最后一个周期
                if current_period:
                    abortion_periods.append(current_period)

                # 计算每个流产周期的持续天数
                if abortion_periods:
                    durations = [(period[-1] - period[0]).days + 1 for period in abortion_periods]
                    avg_duration = sum(durations) / len(durations)

                    # 更新该猪场的指标
                    farm_data[pigfarm][f'{period}_num_abortion_periods'] = len(abortion_periods)
                    farm_data[pigfarm][f'{period}_avg_duration_days'] = avg_duration

        # 转换结果为DataFrame
        result_df = pd.DataFrame(list(farm_data.values()))

        # 计算所有指标的平均值
        if not result_df.empty:
            avg_data = {
                'pigfarm_dk': '整体平均值',
                '1_7_num_abortion_periods': result_df['1_7_num_abortion_periods'].mean(),
                '1_7_avg_duration_days': result_df['1_7_avg_duration_days'].mean(),
                '8_14_num_abortion_periods': result_df['8_14_num_abortion_periods'].mean(),
                '8_14_avg_duration_days': result_df['8_14_avg_duration_days'].mean(),
                '15_21_num_abortion_periods': result_df['15_21_num_abortion_periods'].mean(),
                '15_21_avg_duration_days': result_df['15_21_avg_duration_days'].mean()
            }

            # 在DataFrame的开头添加平均值行
            result_df = pd.concat([pd.DataFrame([avg_data]), result_df], ignore_index=True)

        return result_df

    # 计算每个猪场流产率异常的时间间隔
    def calculate_abortion_interval(self):
        """
        计算每个猪场的流产率超过阈值间隔时间,间隔时间定义:以开始时间为基准，计算每个猪场的流产间隔时间
        :return: DataFrame 包含每个样本的流产间隔时间
        """
        ground_truth = self.predict_data.copy()

        # 创建结果字典，用于存储每个猪场的数据
        farm_data = {}

        # 按猪场分组
        for pigfarm, group_data in ground_truth.groupby('pigfarm_dk'):
            farm_data[pigfarm] = {
                'pigfarm_dk': pigfarm,
                '1_7_num_intervals': 0,
                '1_7_avg_interval_days': 0,
                '8_14_num_intervals': 0,
                '8_14_avg_interval_days': 0,
                '15_21_num_intervals': 0,
                '15_21_avg_interval_days': 0
            }

            # 处理每种流产周期类型
            for period in ['1_7', '8_14', '15_21']:
                # 1. 筛选出abort_{period}为1的样本
                if self.truth_column.format(period) not in group_data.columns:
                    continue

                abortion_samples = group_data[group_data[self.truth_column.format(period)] == 1].sort_values('stats_dt')

                if len(abortion_samples) == 0:
                    continue

                # 2. 将样本按照时间是否连续分成多个时间段
                abortion_periods = []
                current_period = [abortion_samples.iloc[0]['stats_dt']]

                for i in range(1, len(abortion_samples)):
                    curr_date = abortion_samples.iloc[i]['stats_dt']
                    prev_date = abortion_samples.iloc[i-1]['stats_dt']

                    # 判断是否连续（日期相差1天）
                    if (curr_date - prev_date).days == 1:
                        # 连续，添加到当前周期
                        current_period.append(curr_date)
                    else:
                        # 不连续，结束当前周期并开始新周期
                        abortion_periods.append(current_period)
                        current_period = [curr_date]

                # 添加最后一个周期
                if current_period:
                    abortion_periods.append(current_period)

                # 3. 计算每个时间段的开头日期之差
                if len(abortion_periods) > 1:
                    # 获取每个流产周期的开始日期
                    start_dates = [period[0] for period in abortion_periods]

                    # 计算相邻开始日期之间的间隔
                    intervals = [(start_dates[i+1] - start_dates[i]).days for i in range(len(start_dates)-1)]

                    # 4. 求平均值
                    avg_interval = sum(intervals) / len(intervals)

                    # 更新该猪场的指标
                    farm_data[pigfarm][f'{period}_num_intervals'] = len(intervals)
                    farm_data[pigfarm][f'{period}_avg_interval_days'] = avg_interval

        # 转换结果为DataFrame
        result_df = pd.DataFrame(list(farm_data.values()))

        # 计算所有指标的平均值
        if not result_df.empty:
            avg_data = {
                'pigfarm_dk': '整体平均值',
                '1_7_num_intervals': result_df['1_7_num_intervals'].mean(),
                '1_7_avg_interval_days': result_df['1_7_avg_interval_days'].mean(),
                '8_14_num_intervals': result_df['8_14_num_intervals'].mean(),
                '8_14_avg_interval_days': result_df['8_14_avg_interval_days'].mean(),
                '15_21_num_intervals': result_df['15_21_num_intervals'].mean(),
                '15_21_avg_interval_days': result_df['15_21_avg_interval_days'].mean()
            }

            # 在DataFrame的开头添加平均值行
            result_df = pd.concat([pd.DataFrame([avg_data]), result_df], ignore_index=True)

        return result_df


