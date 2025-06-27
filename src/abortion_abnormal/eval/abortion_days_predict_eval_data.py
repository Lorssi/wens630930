import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, classification_report, cohen_kappa_score, confusion_matrix, mean_absolute_error, mean_squared_error
from sklearn.exceptions import UndefinedMetricWarning
from tqdm import tqdm
import numpy as np
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

class AbortionDaysPredictEvalData():

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
        self.logger = logger
        self.feiwen_dict = None
        self.truth_column = 'abort_days_{}'
        self.predict_column = 'abort_days_{}_decision'
        self.probability_column = 'prob_abort_day_{}_pred_{}'


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


    def calculate_trend_accuracy(self, period, prev_status, next_status):
        """
        计算趋势准确率
        :param period: 预测周期，'1_7', '8_14', '15_21'
        :param prev_status: 前一天的状态，0或1
        :param next_status: 后一天的状态，0或1
        :return: 上升趋势准确率
        """
        data = self.predict_data.copy()

        # 总天数 T1
        total_days = 0
        # 预测正确的天数 T2
        correct_days = 0

        # 按猪场分组
        for pigfarm, group_data in data.groupby('pigfarm_dk'):
            # 按时间排序
            sorted_data = group_data.sort_values('stats_dt')

            # 获取真实值和预测值
            y_true = sorted_data[self.truth_column.format(period)].values
            y_pred = sorted_data[self.predict_column.format(period)].values

            if prev_status == 0 and next_status == 1:
                # 查找真实值递增的天数
                for i in range(1, len(y_true)):
                    if y_true[i] > y_true[i-1]:  # 真实值递增
                        total_days += 1
                        # 检查预测值是否也递增
                        if y_pred[i] > y_pred[i-1]:
                            correct_days += 1
            elif prev_status == 1 and next_status == 0:
                # 查找真实值递减的天数
                for i in range(1, len(y_true)):
                    if y_true[i] < y_true[i-1]:
                        total_days += 1
                        # 检查预测值是否也递减
                        if y_pred[i] < y_pred[i-1]:
                            correct_days += 1

        # 计算趋势准确率
        trend_accuracy = correct_days / total_days if total_days > 0 else 0

        return trend_accuracy


    def get_special_sample(self, data, period, prev_status, next_status):
        """
        获取特殊样本
        :param data: 输入数据 DataFrame
        :param period: 预测周期，'1_7', '8_14', '15_21'
        :param prev_status: 前一天的状态，0或1
        :param next_status: 后一天的状态，0或1
        :return: 特殊样本 DataFrame
        """
        # 特殊样本
        special_samples = []

        # 按照猪场分组
        grouped = data.groupby('pigfarm_dk')

        # 确定前后状态
        if prev_status == 0 and next_status == 1: # 评测从正常到异常的样本
            for pigfarm, group_data in grouped:
                # 按时间排序
                sorted_data = group_data.sort_values('stats_dt')

                # 找出时间t时abort_period为0，时间t+1时abort_period为1的时间点
                for i in range(len(sorted_data) - 1):
                    if sorted_data.iloc[i][self.truth_column.format(period)] == prev_status and sorted_data.iloc[i+1][self.truth_column.format(period)] == next_status:
                        # 获取时间点t
                        t = sorted_data.iloc[i]['stats_dt']
                        # 获取时间点t-1和t+7
                        t_minus_1 = t - pd.Timedelta(days=1)
                        t_plus_7 = t + pd.Timedelta(days=7)
                        # 筛选时间范围内的样本
                        time_range_samples = sorted_data[(sorted_data['stats_dt'] >= t_minus_1) &
                                                        (sorted_data['stats_dt'] <= t_plus_7)]
                        # 创建一个以日期为索引的DataFrame方便查找
                        time_range_samples_index = time_range_samples.set_index('stats_dt')

                        # 正常部分样本，范围为2
                        # 确保t-1的abort_period为0
                        t_minus_1_sample = time_range_samples[time_range_samples['stats_dt'] == t_minus_1]
                        if t_minus_1_sample.empty or t_minus_1_sample.iloc[0][self.truth_column.format(period)] != 0:
                            time_range_samples = time_range_samples[time_range_samples['stats_dt'] != t_minus_1]

                        # 异常部分样本，范围为递增范围
                        current_day = t + pd.Timedelta(days=2)
                        prev_value = 1
                        while current_day in time_range_samples_index.index:
                            current_value = time_range_samples_index.loc[current_day][self.truth_column.format(period)]
                            if current_value < prev_value:
                                break
                            prev_value = current_value
                            current_day += pd.Timedelta(days=1)
                        # 筛选出递增范围内的样本
                        time_range_samples = time_range_samples[(time_range_samples['stats_dt'] < current_day)]
                        special_samples.append(time_range_samples)

        elif prev_status == 1 and next_status == 0: # 评测从异常到正常的样本
            for pigfarm, group_data in grouped:
                # 按时间排序
                sorted_data = group_data.sort_values('stats_dt')

                # 找出时间t时abort_period为1，时间t+1时abort_period为0的时间点
                for i in range(len(sorted_data) - 1):
                    if sorted_data.iloc[i][self.truth_column.format(period)] == prev_status and sorted_data.iloc[i+1][self.truth_column.format(period)] == next_status:
                        # 获取时间点t
                        t = sorted_data.iloc[i]['stats_dt']
                        # 获取时间点t-1和t+7
                        t_minus_6 = t - pd.Timedelta(days=6)
                        t_plus_2 = t + pd.Timedelta(days=2)
                        # 筛选时间范围内的样本
                        time_range_samples = sorted_data[(sorted_data['stats_dt'] >= t_minus_6) &
                                                        (sorted_data['stats_dt'] <= t_plus_2)]
                        # 创建一个以日期为索引的DataFrame方便查找
                        time_range_samples_index = time_range_samples.set_index('stats_dt')

                        # 正常部分样本，范围为2
                        # 确保t+2的abort_period为0
                        t_plus_2_sample = time_range_samples[time_range_samples['stats_dt'] == t_plus_2]
                        if t_plus_2_sample.empty or t_plus_2_sample.iloc[0][self.truth_column.format(period)] != 0:
                            time_range_samples = time_range_samples[time_range_samples['stats_dt'] != t_plus_2]

                        # 异常部分样本，范围为递增范围
                        current_day = t - pd.Timedelta(days=1)
                        prev_value = 1
                        while current_day in time_range_samples_index.index:
                            current_value = time_range_samples_index.loc[current_day][self.truth_column.format(period)]
                            if current_value < prev_value:
                                break
                            prev_value = current_value
                            current_day -= pd.Timedelta(days=1)
                        # 筛选出递增范围内的样本
                        time_range_samples = time_range_samples[(time_range_samples['stats_dt'] > current_day)]

                        special_samples.append(time_range_samples)

        elif prev_status == 1 and next_status == 1: # 评测从异常到异常的样本
            special_samples = data[(data[self.truth_column.format(period)] == 6) | (data[self.truth_column.format(period)] == 7)].copy()
            return special_samples

        elif prev_status == 1 and next_status == 2: # 评测被单个异常流产率的样本
            for pigfarm, group_data in grouped:
                # 按时间排序
                sorted_data = group_data.sort_values('stats_dt')

                # 找出时间t时abort_period为0，时间t+1时abort_period为1的时间点
                for i in range(len(sorted_data) - 1):
                    if sorted_data.iloc[i][self.truth_column.format(period)] == 0 and sorted_data.iloc[i+1][self.truth_column.format(period)] == 1:
                        # 获取时间点t
                        t = sorted_data.iloc[i]['stats_dt']
                        # 获取时间点t-1和t+7
                        t_minus_1 = t - pd.Timedelta(days=1)
                        t_plus_7 = t + pd.Timedelta(days=7)
                        # 筛选时间范围内的样本
                        time_range_samples = sorted_data[(sorted_data['stats_dt'] >= t_minus_1) &
                                                        (sorted_data['stats_dt'] <= t_plus_7)]
                        # 创建一个以日期为索引的DataFrame方便查找
                        time_range_samples_index = time_range_samples.set_index('stats_dt')

                        # 正常部分样本，范围为2
                        # 确保t-1的abort_period为0
                        t_minus_1_sample = time_range_samples[time_range_samples['stats_dt'] == t_minus_1]
                        if t_minus_1_sample.empty or t_minus_1_sample.iloc[0][self.truth_column.format(period)] != 0:
                            time_range_samples = time_range_samples[time_range_samples['stats_dt'] != t_minus_1]

                        # 异常部分样本，范围为递增范围
                        current_day = t + pd.Timedelta(days=2)
                        prev_value = 1
                        while current_day in time_range_samples_index.index:
                            current_value = time_range_samples_index.loc[current_day][self.truth_column.format(period)]
                            if current_value < prev_value:
                                break
                            prev_value = current_value
                            current_day += pd.Timedelta(days=1)
                        # 筛选出递增范围内的样本
                        time_range_samples = time_range_samples[(time_range_samples['stats_dt'] < current_day)]

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

        else:
            raise ValueError("prev_status 和 next_status 的组合不正确，请检查输入参数。")

        # 合并所有特殊样本
        if len(special_samples) == 0:
            self.logger.warning(f"没有找到满足条件的特殊样本，abortion_period: {period}, prev_status: {prev_status}, next_status: {next_status}")
            return pd.DataFrame()

        return pd.concat(special_samples)


    def special_sample_1_eval_one_periods_metric(self, period, prev_status, next_status):
        """
        特殊样本1：计算前一天为prev_status(01)，后一天为next_status(10)的评估指标
        :param period: 预测周期，'1_7', '8_14', '15_21'
        :param prev_status: 前一天的状态，0或1
        :param next_status: 后一天的状态，0或1
        """
        if prev_status == 0 and next_status == 1:
            sample_type = 'normal-2_to_abnormal'
        elif prev_status == 1 and next_status == 0:
            sample_type = 'abnormal_to_normal-2'
        elif prev_status == 1 and next_status == 2:
            sample_type = 'normal-2_to_abnormal-2_single_influence'

        # 1. 获取特殊样本
        special_samples = self.get_special_sample(self.predict_data.copy(), period, prev_status, next_status)
        # 2. 去重
        special_samples = special_samples.drop_duplicates()
        # 计算原始样本数量
        total_samples_num = len(special_samples)
        # 3. 剔除非瘟数据
        special_samples = self.exclude_feiwen_data(special_samples, period)
        # 计算剔除后的样本数量
        filtered_samples_num = len(special_samples)

        # 4. 计算评估指标
        y_true = special_samples[self.truth_column.format(period)].values
        y_pred = special_samples[self.predict_column.format(period)].values

        # 计算指标
        report = classification_report(y_true, y_pred, labels=list(range(8)), output_dict=True, zero_division=0)
        trend_accuracy = self.calculate_trend_accuracy(period, prev_status, next_status)
        kappa = cohen_kappa_score(y_true, y_pred, labels=list(range(8)))
        conf_matrix = confusion_matrix(y_true, y_pred, labels=list(range(8)))
        overall_mae = mean_absolute_error(y_true, y_pred)
        overall_rmse = np.sqrt(mean_squared_error(y_true, y_pred))

        # 初始化结果列表
        results = []
        valid_auc_scores = []

        # 为每个类别创建一条记录
        for label in range(8):
            if str(label) in report:
                # 计算当前类别的MAE和RMSE
                # 筛选出真实标签为当前label的样本
                label_mask = (y_true == label)
                if np.sum(label_mask) > 0:  # 确保有该标签的样本
                    y_true_label = y_true[label_mask]
                    y_pred_label = y_pred[label_mask]
                    label_mae = mean_absolute_error(y_true_label, y_pred_label)
                    label_rmse = np.sqrt(mean_squared_error(y_true_label, y_pred_label))
                else:
                    label_mae = None
                    label_rmse = None
                # 计算当前类别的ovr auc
                auc_score = None
                try:
                    # 创建二元标签
                    y_true_ovr = (y_true == label).astype(int)
                    # 获取当前类别的预测概率
                    y_prob_ovr = special_samples[self.probability_column.format(period, label)].values
                    with warnings.catch_warnings():
                        warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
                        if len(np.unique(y_true_ovr)) > 1:  # 确保有正负样本
                            auc_score = roc_auc_score(y_true_ovr, y_prob_ovr)
                            valid_auc_scores.append(auc_score)
                except Exception as e:
                    self.logger.warning(f"无法计算类别{label}的AUC: {e}")
                result_row = {
                    'stats_dt': f"{self.eval_running_dt_start_string} ~ {self.eval_running_dt_end_string}",
                    'sample_type': sample_type,
                    'total_sample_num': total_samples_num,
                    'remain_sample_num': filtered_samples_num,
                    'eval_period': period,
                    'label': label,
                    'precision': report[str(label)]['precision'],
                    'recall': report[str(label)]['recall'],
                    'f1_score': report[str(label)]['f1-score'],
                    'auc': auc_score,
                    'kappa': kappa,
                    'MAE': label_mae,
                    'RMSE': label_rmse,
                    'trend_accuracy': trend_accuracy,
                    'recognition': self.recognition
                }
                results.append(result_row)

        # 计算加权指标
        result_row = {
            'stats_dt': f"{self.eval_running_dt_start_string} ~ {self.eval_running_dt_end_string}",
            'sample_type': sample_type,
            'total_sample_num': total_samples_num,
            'remain_sample_num': filtered_samples_num,
            'eval_period': period,
            'label': 'all',
            'precision': report['weighted avg']['precision'],
            'recall': report['weighted avg']['recall'],
            'f1_score': report['weighted avg']['f1-score'],
            'auc': np.mean(valid_auc_scores) if valid_auc_scores else None,
            'kappa': kappa,
            'MAE': overall_mae,
            'RMSE': overall_rmse,
            'trend_accuracy': trend_accuracy,
            'recognition': self.recognition
        }
        results.append(result_row)

        # 创建DataFrame并指定列顺序
        result_df = pd.DataFrame(results)
        column_order = ['stats_dt', 'sample_type', 'total_sample_num', 'remain_sample_num',
                    'eval_period', 'label', 'precision', 'recall', 'f1_score', 'auc',
                    'kappa', 'MAE', 'RMSE', 'trend_accuracy', 'recognition']
        result_df = result_df[column_order]

        self.result = pd.concat([self.result, pd.DataFrame(results)], ignore_index=True)

        return conf_matrix


    def special_sample_2_eval_one_periods_metric(self, period):
        """
        特殊样本2：计算abnormal_to_abnormal的评估指标
        :param period: 预测周期，'1_7', '8_14', '15_21'
        """
        # 1. 获取特殊样本
        special_samples = self.get_special_sample(self.predict_data.copy(), period, 1, 1)
        # 2. 去重
        special_samples = special_samples.drop_duplicates()
        # 计算原始样本数量
        total_samples_num = len(special_samples)
        # 3. 剔除非瘟数据
        special_samples = self.exclude_feiwen_data(special_samples, period)
        # 计算剔除后的样本数量
        filtered_samples_num = len(special_samples)

        # 4. 计算评估指标
        y_true = special_samples[self.truth_column.format(period)].values
        y_pred = special_samples[self.predict_column.format(period)].values
        # 计算指标
        report = classification_report(y_true, y_pred, labels=list(range(8)), output_dict=True, zero_division=0)
        kappa = cohen_kappa_score(y_true, y_pred)
        conf_matrix = confusion_matrix(y_true, y_pred, labels=list(range(8)))
        overall_mae = mean_absolute_error(y_true, y_pred)
        overall_rmse = np.sqrt(mean_squared_error(y_true, y_pred))

        # 初始化结果列表
        results = []
        valid_auc_scores = []
        # 为每个类别创建一条记录
        for label in range(0, 8):
            if str(label) in report:
                # 计算当前类别的MAE和RMSE
                # 筛选出真实标签为当前label的样本
                label_mask = (y_true == label)
                if np.sum(label_mask) > 0:  # 确保有该标签的样本
                    y_true_label = y_true[label_mask]
                    y_pred_label = y_pred[label_mask]
                    label_mae = mean_absolute_error(y_true_label, y_pred_label)
                    label_rmse = np.sqrt(mean_squared_error(y_true_label, y_pred_label))
                else:
                    label_mae = None
                    label_rmse = None
                # 计算当前类别的ovr auc
                auc_score = None
                try:
                    # 创建二元标签
                    y_true_ovr = (y_true == label).astype(int)
                    # 获取当前类别的预测概率
                    y_prob_ovr = special_samples[self.probability_column.format(period, label)].values
                    with warnings.catch_warnings():
                        warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
                        if len(np.unique(y_true_ovr)) > 1:  # 确保有正负样本
                            auc_score = roc_auc_score(y_true_ovr, y_prob_ovr)
                            valid_auc_scores.append(auc_score)
                except Exception as e:
                    self.logger.warning(f"无法计算类别{label}的AUC: {e}")
                result_row = {
                    'stats_dt': f"{self.eval_running_dt_start_string} ~ {self.eval_running_dt_end_string}",
                    'sample_type': 'abnormal_to_abnormal',
                    'total_sample_num': total_samples_num,
                    'remain_sample_num': filtered_samples_num,
                    'eval_period': period,
                    'label': label,
                    'precision': report[str(label)]['precision'],
                    'recall': report[str(label)]['recall'],
                    'f1_score': report[str(label)]['f1-score'],
                    'auc': auc_score,
                    'kappa': kappa,
                    'MAE': label_mae,
                    'RMSE': label_rmse,
                    'trend_accuracy': '-',
                    'recognition': self.recognition
                }

                results.append(result_row)

        # 计算加权指标
        result_row = {
            'stats_dt': f"{self.eval_running_dt_start_string} ~ {self.eval_running_dt_end_string}",
            'sample_type': 'abnormal_to_abnormal',
            'total_sample_num': total_samples_num,
            'remain_sample_num': filtered_samples_num,
            'eval_period': period,
            'label': 'all',
            'precision': report['weighted avg']['precision'],
            'recall': report['weighted avg']['recall'],
            'f1_score': report['weighted avg']['f1-score'],
            'auc': np.mean(valid_auc_scores) if valid_auc_scores else None,
            'kappa': kappa,
            'MAE': overall_mae,
            'RMSE': overall_rmse,
            'trend_accuracy': '-',
            'recognition': self.recognition
        }

        results.append(result_row)
        self.result = pd.concat([self.result, pd.DataFrame(results)], ignore_index=True)

        return conf_matrix


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

        # 计算剔除后的样本数量
        filtered_samples_num = len(data)

        # 获取标签和预测值
        y_true = data[self.truth_column.format(period)].values
        y_pred = data[self.predict_column.format(period)].values

        # 计算指标
        report = classification_report(y_true, y_pred, labels=list(range(8)), output_dict=True, zero_division=0)
        # 计算kappa系数和混合矩阵
        # 获取实际存在的标签
        unique_labels = np.unique(np.concatenate([y_true, y_pred]))
        valid_labels = [label for label in range(8) if label in unique_labels]
        # 如果有效标签少于2个，使用所有可能的标签
        labels_for_metrics = valid_labels if len(valid_labels) >= 2 else list(range(8))
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            try:
                kappa = cohen_kappa_score(y_true, y_pred, labels=labels_for_metrics)
                # 检查kappa是否为NaN或无穷大
                if np.isnan(kappa) or np.isinf(kappa):
                    kappa = 0.0
            except Exception as e:
                self.logger.warning(f"计算kappa系数时出错: {e}")
                kappa = 0.0
        conf_matrix = confusion_matrix(y_true, y_pred, labels=list(range(8)))
        # 计算整体MAE和RMSE
        overall_mae = mean_absolute_error(y_true, y_pred)
        overall_rmse = np.sqrt(mean_squared_error(y_true, y_pred))

        # 初始化结果列表
        results = []

        # 用于收集所有有效的AUC分数
        valid_auc_scores = []

        # 为每个类别创建一条记录
        for label in range(8):
            if str(label) in report:
                # 计算当前类别的MAE和RMSE
                # 筛选出真实标签为当前label的样本
                label_mask = (y_true == label)
                if np.sum(label_mask) > 0:  # 确保有该标签的样本
                    y_true_label = y_true[label_mask]
                    y_pred_label = y_pred[label_mask]
                    label_mae = mean_absolute_error(y_true_label, y_pred_label)
                    label_rmse = np.sqrt(mean_squared_error(y_true_label, y_pred_label))
                else:
                    label_mae = None
                    label_rmse = None
                # 计算当前类别的ovr auc
                auc_score = None
                try:
                    # 创建二元标签
                    y_true_ovr = (y_true == label).astype(int)
                    # 获取当前类别的预测概率
                    y_prob_ovr = data[self.probability_column.format(period, label)].values
                    with warnings.catch_warnings():
                        warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
                        if len(np.unique(y_true_ovr)) > 1:  # 确保有正负样本
                            auc_score = roc_auc_score(y_true_ovr, y_prob_ovr)
                            valid_auc_scores.append(auc_score)
                except Exception as e:
                    self.logger.warning(f"无法计算类别{label}的AUC: {e}")

                result_row = {
                    'stats_dt': f"{self.eval_running_dt_start_string} ~ {self.eval_running_dt_end_string}",
                    f'l{level}_name': name,
                    'total_sample_num': total_samples_num,
                    'remain_sample_num': filtered_samples_num,
                    'eval_period': period,
                    'label': label,
                    'precision': report[str(label)]['precision'],
                    'recall': report[str(label)]['recall'],
                    'f1_score': report[str(label)]['f1-score'],
                    'auc': auc_score,
                    'kappa': kappa,
                    'MAE': label_mae,
                    'RMSE': label_rmse,
                    'recognition': self.recognition
                }
                results.append(result_row)

        # 计算加权指标
        result_row = {
            'stats_dt': f"{self.eval_running_dt_start_string} ~ {self.eval_running_dt_end_string}",
            f'l{level}_name': name,
            'total_sample_num': total_samples_num,
            'remain_sample_num': filtered_samples_num,
            'eval_period': period,
            'label': 'all',
            'precision': report['weighted avg']['precision'],
            'recall': report['weighted avg']['recall'],
            'f1_score': report['weighted avg']['f1-score'],
            'auc': np.mean(valid_auc_scores) if valid_auc_scores else None,
            'kappa': kappa,
            'MAE': overall_mae,
            'RMSE': overall_rmse,
            'recognition': self.recognition
        }
        results.append(result_row)

        result_df = pd.DataFrame(results)
        if not exclude_feiwen:
            result_df.drop(columns=['remain_sample_num'], inplace=True)
        if hierarchical_data is None:
            result_df.drop(columns=[f'l{level}_name'], inplace=True)

        self.result = pd.concat([self.result, result_df], ignore_index=True)

        return conf_matrix


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
        conf_matrix_list = {}  # 用于存储每个周期的混淆矩阵
        for abortion_period in tqdm(['1_7', '8_14', '15_21'], desc='计算剔除非瘟整体指标'):
            conf_matrix = self.overall_eval_one_periods_metric(abortion_period)
            conf_matrix_list[abortion_period] = (conf_matrix)

        return self.result, conf_matrix_list

    # 计算特殊样本指标
    def eval_special_samples(self):
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
            return self.result

    # 计算分级组织指标
    def eval_organizational_hierarchy(self):
        l2_results, l3_results, l4_results = self.eval_with_organizational_hierarchy()
        return l2_results, l3_results, l4_results

       # 计算每个猪场的流产持续时间数量和持续时间平均值

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

    def calculate_label_num(self):
        """
        计算每个标签的样本数量
        :return: DataFrame 包含每个标签的样本数量
        """
        ground_truth = self.predict_data.copy()

        # 创建结果字典，用于存储每个标签的数据
        label_data = {}

        # 处理每种流产周期类型
        for period in ['1_7', '8_14', '15_21']:
            if self.truth_column.format(period) not in ground_truth.columns:
                continue

            # 获取当前周期的标签列
            label_column = self.truth_column.format(period)

            # 计算每个标签的样本数量
            label_counts = ground_truth[label_column].value_counts().to_dict()

            # 将结果存入字典
            for label, count in label_counts.items():
                if label not in label_data:
                    label_data[label] = {'label': label}
                label_data[label][f'{period}_count'] = count

        # 转换结果为DataFrame
        result_df = pd.DataFrame.from_dict(label_data, orient='index').reset_index(drop=True)
        result_df.sort_values(by='label', inplace=True)

        return result_df