import pandas as pd
import logging
import os
import sys
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, classification_report
from sklearn.exceptions import UndefinedMetricWarning
from tqdm import tqdm
import warnings

from abortion_abnormal.eval.eval_base import EvalBaseMixin

# base_dir = os.path.split(os.path.realpath(__file__))[0]
program = os.path.basename(sys.argv[0])
logger = logging.getLogger(program)
logging.basicConfig(format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s: %(message)s',
                    datefmt='%a, %d %b %Y %H:%M:%S')
logging.root.setLevel(level=logging.INFO)
# warnings.filterwarnings("ignore")

# 任务1评估
class AbortionAbnormalEval1(EvalBaseMixin):
    def __init__(self, logger=None):
        self.logger = logger
        self.result = pd.DataFrame()

    # 构建评测数据集
    def build_eval_set(self, eval_running_dt_end, eval_interval, param=None):
        self.eval_running_dt_end = eval_running_dt_end
        eval_running_dt_end = pd.to_datetime(eval_running_dt_end)
        self.eval_start_dt = (eval_running_dt_end - pd.Timedelta(days=eval_interval)).strftime('%Y-%m-%d')
        # 获取预测结果文件
        '''
        stats_dt, pigfarm_dk 运行日期，猪场id
        abortion_1_7, abortion_8_14, abortion_15_21 真实值，01
        predicted_abortion_1_7, predicted_abortion_8_14, predicted_abortion_15_21 预测值，01
        probability_abortion_1_7, probability_abortion_8_14, probability_abortion_15_21 # 预测概率
        '''
        self.predict_data = pd.read_csv("data/predict/has_risk_predict_result.csv", encoding='utf-8')
        # 转化日期格式
        self.predict_data['stats_dt'] = pd.to_datetime(self.predict_data['stats_dt'])

        # 获取组织分类表
        dim_org_inv = pd.read_csv("data/raw_data/dim_org_inv.csv", encoding='utf-8')

        # 处理组织分类信息，只保留需要的列
        org_mapping = dim_org_inv[['org_inv_dk', 'l2_org_inv_nm', 'l3_org_inv_nm', 'l4_org_inv_nm']].copy()

        # 确保ID列为字符串类型以避免join时的类型不匹配问题
        org_mapping['org_inv_dk'] = org_mapping['org_inv_dk'].astype(str)
        self.predict_data['pigfarm_dk'] = self.predict_data['pigfarm_dk'].astype(str)
        
        # 将预测数据与组织数据合并，添加部门信息
        self.predict_data = self.predict_data.merge(
            org_mapping,
            left_on='pigfarm_dk',
            right_on='org_inv_dk',
            how='left'
        )



    def get_eval_index_sample(self):
        return self.index_sample


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
        
        # 获取非瘟数据
        feiwen_data = pd.read_csv("data/raw_data/ADS_PIG_FARM_AND_REARER_ONSET.csv", encoding='utf-8')
        # 只保留需要的列
        feiwen_data = feiwen_data[['org_inv_dk', 'start_dt', 'end_dt']]
        # 转换日期格式
        feiwen_data['start_dt'] = pd.to_datetime(feiwen_data['start_dt'])
        feiwen_data['end_dt'] = pd.to_datetime(feiwen_data['end_dt'])
        
        # 确保ID为字符串类型
        sample = sample.copy()
        sample['pigfarm_dk'] = sample['pigfarm_dk'].astype(str)
        feiwen_data['org_inv_dk'] = feiwen_data['org_inv_dk'].astype(str)
        
        # 计算样本的预测开始和结束日期
        sample['predict_start'] = sample['stats_dt'] + pd.Timedelta(days=start)
        sample['predict_end'] = sample['stats_dt'] + pd.Timedelta(days=end)
        
        # 按猪场ID对非瘟数据进行分组
        feiwen_dict = {}
        for farm_id, group in feiwen_data.groupby('org_inv_dk'):
            feiwen_dict[farm_id] = group[['start_dt', 'end_dt']].values.tolist()
        
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


    def special_sample_1_calculate_special_recall(self, period, prev_status, next_status):
        """
        计算特殊的召回率指标
        按时间段粒度计算：先计算每个特殊时间段的recall，再计算每个猪场的平均recall，最后计算所有猪场的平均recall
        """
        # 获取原始数据
        data = self.predict_data.copy()

        # 保存每个猪场的recall值
        farm_recalls = {}
        transition_recalls = []
        
        # 按照猪场分组
        grouped = data.groupby('pigfarm_dk')

        for pigfarm, group_data in grouped:
            # 按时间排序
            sorted_data = group_data.sort_values('stats_dt')
            farm_transitions_recalls = []
    
            # 找出满足条件的时间点
            for i in range(len(sorted_data) - 1):
                if sorted_data.iloc[i][f'abortion_{period}'] == prev_status and sorted_data.iloc[i+1][f'abortion_{period}'] == next_status:
                    t = sorted_data.iloc[i]['stats_dt']
                    
                    # 找出t-1到t+2的所有样本
                    farm_data = sorted_data.copy()
                    t_minus_1 = t - pd.Timedelta(days=1)
                    t_plus_2 = t + pd.Timedelta(days=2)
                    
                    # 筛选时间范围内的样本
                    time_range_samples = farm_data[(farm_data['stats_dt'] >= t_minus_1) & 
                                                  (farm_data['stats_dt'] <= t_plus_2)]
                    
                    # 去重并剔除非瘟数据
                    time_range_samples = time_range_samples.drop_duplicates()
                    time_range_samples = self.exclude_feiwen_data(time_range_samples, period)
                    
                    # 如果剔除后还有样本
                    if len(time_range_samples) > 0:
                        # 计算这个时间段的recall
                        y_true = time_range_samples[f'abortion_{period}'].values
                        y_pred = time_range_samples[f'predicted_abortion_{period}'].values
                        
                        # 只有当实际有阳性样本时才能计算recall
                        if sum(y_true) > 0:
                            transition_recall = recall_score(y_true, y_pred, zero_division=0)
                            farm_transitions_recalls.append(transition_recall)
                            transition_recalls.append(transition_recall)
            
            # 计算该猪场的平均recall
            if farm_transitions_recalls:
                farm_recalls[pigfarm] = sum(farm_transitions_recalls) / len(farm_transitions_recalls)
        
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
        # 1. 筛选特殊样本
        special_samples = []
        # 获取预测数据
        data = self.predict_data.copy()
        
        # 按照猪场分组
        grouped = data.groupby('pigfarm_dk')

        for pigfarm, group_data in grouped:
            # 按时间排序
            sorted_data = group_data.sort_values('stats_dt')

            # 找出时间t时abortion_period为0，时间t+1时abortion_period为1的时间点
            for i in range(len(sorted_data) - 1):
                if sorted_data.iloc[i][f'abortion_{period}'] == prev_status and sorted_data.iloc[i+1][f'abortion_{period}'] == next_status:
                    t = sorted_data.iloc[i]['stats_dt']

                    # 找出t-1到t+2的所有样本
                    farm_data = sorted_data.copy()
                    t_minus_1 = t - pd.Timedelta(days=1)
                    t_plus_2 = t + pd.Timedelta(days=2)
                    
                    # 筛选时间范围内的样本
                    time_range_samples = farm_data[(farm_data['stats_dt'] >= t_minus_1) & 
                                                  (farm_data['stats_dt'] <= t_plus_2)]
                    
                    # 添加到特殊样本列表
                    special_samples.append(time_range_samples)
        
        # 合并所有特殊样本
        if len(special_samples) > 0:
            self.index_sample = pd.concat(special_samples)
            # 2. 去重
            self.index_sample = self.index_sample.drop_duplicates()
            # 计算原始样本数量
            total_samples_num = len(self.index_sample)
            # 3. 剔除非瘟数据
            self.index_sample = self.exclude_feiwen_data(self.index_sample, period)
            # 计算剔除后的样本数量
            filtered_samples_num = len(self.index_sample)
        else:
            self.logger.warning(f"没有找到满足条件的特殊样本，abortion_period: {period}")
            return 
        
        # 3. 计算评估指标
        y_true = self.index_sample[f'abortion_{period}'].values
        y_pred = self.index_sample[f'predicted_abortion_{period}'].values
        y_prob = self.index_sample[f'probability_abortion_{period}'].values
        
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
        special_recall = self.special_sample_1_calculate_special_recall(period, prev_status, next_status)

        # 保存结果
        result_row = {
            'stats_dt': f"{self.eval_start_dt} ~ {self.eval_running_dt_end}",
            'sample_type': sample_type,
            'total_sample_num': total_samples_num,
            'remain_sample_num': filtered_samples_num,
            'eval_period': period,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'auc': auc,
            'special_recall': special_recall,
        }
        self.result = pd.concat([self.result, pd.DataFrame([result_row])], ignore_index=True)
        

    def special_sample_2_calculate_special_recall(self, period):
        """
        计算特殊的召回率指标 - 针对所有真实值为1的样本
        按时间段粒度计算：先将标签为1的样本按时间连续性分组，计算每个时间段的recall
        再计算每个猪场的平均recall，最后计算所有猪场的平均recall
        :param period: 预测周期，'1_7', '8_14', '15_21'
        """
        # 获取原始数据
        data = self.predict_data.copy()
        
        # 筛选真实值为1的样本
        positive_samples = data[data[f'abortion_{period}'] == 1].copy()
        
        # 排除非瘟数据
        positive_samples = self.exclude_feiwen_data(positive_samples, period)
        
        # 如果没有符合条件的样本，返回0
        if len(positive_samples) == 0:
            if self.logger:
                self.logger.warning(f"没有找到符合条件的样本，abortion_period: {period}")
            return 0
        
        # 保存每个猪场的recall值
        farm_recalls = {}
        
        # 按照猪场分组
        grouped = positive_samples.groupby('pigfarm_dk')
        
        for pigfarm, group_data in grouped:
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
                y_true = period_data[f'abortion_{period}'].values
                y_pred = period_data[f'predicted_abortion_{period}'].values
                
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


    def special_sample_2_eval_one_periods_metric(self, period):
        """
        特殊样本2：计算真实结果为1
        :param period: 预测周期，'1_7', '8_14', '15_21'
        """
        sample_type = 'abnormal_to_abnormal'
        
        # 1. 筛选特殊样本 - 所有真实值为1的样本
        data = self.predict_data.copy()
        self.index_sample = data[data[f'abortion_{period}'] == 1]
        
        # 计算原始样本数量
        total_samples_num = len(self.index_sample)
        
        if total_samples_num == 0:
            if self.logger:
                self.logger.warning(f"没有找到真实标签为1的样本，abortion_period: {period}")
            return
        
        # 2. 去重
        self.index_sample = self.index_sample.drop_duplicates()
        
        # 3. 剔除非瘟数据
        self.index_sample = self.exclude_feiwen_data(self.index_sample, period)
        
        # 计算剔除后的样本数量
        filtered_samples_num = len(self.index_sample)
        
        # 4. 计算评估指标
        y_true = self.index_sample[f'abortion_{period}'].values
        y_pred = self.index_sample[f'predicted_abortion_{period}'].values
        y_prob = self.index_sample[f'probability_abortion_{period}'].values
        
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
        special_recall = self.special_sample_2_calculate_special_recall(period)
        
        # 保存结果
        result_row = {
            'stats_dt': f"{self.eval_start_dt} ~ {self.eval_running_dt_end}",
            'sample_type': sample_type,
            'total_sample_num': total_samples_num,
            'remain_sample_num': filtered_samples_num,
            'eval_period': period,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'auc': auc,
            'special_recall': special_recall
        }
        self.result = pd.concat([self.result, pd.DataFrame([result_row])], ignore_index=True)


    def overall_eval_one_periods_metric(self, abortion_period, exclude_feiwen=True, hierarchical_data=None, level=None, name=None):
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
            data = self.exclude_feiwen_data(data, abortion_period)

        # 计算剔除后的样本数量
        filtered_samples_num = len(data)

        # 获取标签和预测值
        y_true = data[f'abortion_{abortion_period}'].values
        y_pred = data[f'predicted_abortion_{abortion_period}'].values
        y_prob = data[f'probability_abortion_{abortion_period}'].values

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
                self.logger.warning(f"无法计算{abortion_period}的AUC，可能是样本中只存在一个类别")
        
        # 保存结果
        if name is not None:
            if level == 4:
                print(name)
                print(y_true)
                print(y_pred)
            result_row = {
                'stats_dt': f"{self.eval_start_dt} ~ {self.eval_running_dt_end}",
                f'l{level}_name': name,
                'total_sample_num': total_samples_num,
                'remain_sample_num': filtered_samples_num,
                'eval_period': abortion_period,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'auc': auc
            }
        elif exclude_feiwen:
            result_row = {
                'stats_dt': f"{self.eval_start_dt} ~ {self.eval_running_dt_end}",
                'total_sample_num': total_samples_num,
                'remain_sample_num': filtered_samples_num,
                'eval_period': abortion_period,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'auc': auc
            }
        else:
            result_row = {
                'stats_dt': f"{self.eval_start_dt} ~ {self.eval_running_dt_end}",
                'total_sample_num': total_samples_num,
                'eval_period': abortion_period,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'auc': auc
            }
        self.result = pd.concat([self.result, pd.DataFrame([result_row])], ignore_index=True)


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


    def eval_with_index_sample(self):
        with pd.ExcelWriter(f"data/eval/abortion_abnormal_eval_{self.eval_running_dt_end}.xlsx") as writer:
            # 计算整体指标
            self.result = pd.DataFrame() # 清空上一次的结果
            for abortion_period in tqdm(['1_7', '8_14', '15_21'], desc='计算整体指标'):
                self.overall_eval_one_periods_metric(abortion_period, exclude_feiwen=False)
            self.result.to_excel(writer, sheet_name='整体', index=False)

            # 计算剔除非瘟整体指标
            self.result = pd.DataFrame() # 清空上一次的结果
            for abortion_period in tqdm(['1_7', '8_14', '15_21'], desc='计算剔除非瘟整体指标'):
                self.overall_eval_one_periods_metric(abortion_period)
            self.result.to_excel(writer, sheet_name='整体-剔除非瘟数据', index=False)

            # 计算特殊样本指标
            ## 计算abnormal-2_to_normal-2和normal-2_to_abnormal-2指标
            self.result = pd.DataFrame() # 清空上一次的结果
            for prev_status, next_status in [(0, 1), (1, 0)]:
                for abortion_period in tqdm(['1_7', '8_14', '15_21'], desc=f'计算{prev_status}到{next_status}特殊样本1指标'):
                    # 评估从低到高的指标
                    self.special_sample_1_eval_one_periods_metric(abortion_period, prev_status, next_status)
            ## 计算abnormal_to_abnormal指标
            for abortion_period in tqdm(['1_7', '8_14', '15_21'], desc='计算特殊样本2指标'):
                self.special_sample_2_eval_one_periods_metric(abortion_period)
            ## 保存结果到excel
            self.result.to_excel(writer, sheet_name='特殊样本分析', index=False)

            # 计算分级组织指标
            l2_results, l3_results, l4_results = self.eval_with_organizational_hierarchy()
            l2_results.to_excel(writer, sheet_name='二级组织分类', index=False)
            l3_results.to_excel(writer, sheet_name='三级组织分类', index=False)
            l4_results.to_excel(writer, sheet_name='四级组织分类', index=False)

# 任务2评估
class AbortionAbnormalEval2(EvalBaseMixin):
    pass

# 任务3评估
class AbortionAbnormalEval3(EvalBaseMixin):
    pass