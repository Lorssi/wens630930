import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from sklearn.exceptions import UndefinedMetricWarning
from tqdm import tqdm
import warnings




class EvalData():

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
        self.eval_running_dt_end = pd.to_datetime(eval_running_dt_end)
        self.logger = logger

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
                if sorted_data.iloc[i][f'abort_{period}'] == prev_status and sorted_data.iloc[i+1][f'abort_{period}'] == next_status:
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
                        y_true = time_range_samples[f'abort_{period}'].values
                        y_pred = time_range_samples[f'abort_{period}_decision'].values
                        
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

            # 找出时间t时abort_period为0(1)，时间t+1时abort_period为1(0)的时间点
            for i in range(len(sorted_data) - 1):
                if sorted_data.iloc[i][f'abort_{period}'] == prev_status and sorted_data.iloc[i+1][f'abort_{period}'] == next_status:
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
            index_sample = pd.concat(special_samples)
            # 2. 去重
            index_sample = index_sample.drop_duplicates()
            # 计算原始样本数量
            total_samples_num = len(index_sample)
            # 3. 剔除非瘟数据
            index_sample = self.exclude_feiwen_data(index_sample, period)
            # 计算剔除后的样本数量
            filtered_samples_num = len(index_sample)
        else:
            self.logger.warning(f"没有找到满足条件的特殊样本，abortion_period: {period}")
            return 
        
        # 3. 计算评估指标
        y_true = index_sample[f'abort_{period}'].values
        y_pred = index_sample[f'abort_{period}_decision'].values
        y_prob = index_sample[f'abort_{period}_pred'].values

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
            'stats_dt': f"{self.eval_running_dt_start} ~ {self.eval_running_dt_end}",
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
        positive_samples = data[data[f'abort_{period}'] == 1].copy()
        
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
                y_true = period_data[f'abort_{period}'].values
                y_pred = period_data[f'abort_{period}_decision'].values

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
        index_sample = data[data[f'abort_{period}'] == 1]
        
        # 计算原始样本数量
        total_samples_num = len(index_sample)
        
        if total_samples_num == 0:
            if self.logger:
                self.logger.warning(f"没有找到真实标签为1的样本，abortion_period: {period}")
            return
        
        # 2. 去重
        index_sample = index_sample.drop_duplicates()
        
        # 3. 剔除非瘟数据
        index_sample = self.exclude_feiwen_data(index_sample, period)
        
        # 计算剔除后的样本数量
        filtered_samples_num = len(index_sample)
        
        # 4. 计算评估指标
        y_true = index_sample[f'abort_{period}'].values
        y_pred = index_sample[f'abort_{period}_decision'].values
        y_prob = index_sample[f'abort_{period}_pred'].values

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
            'stats_dt': f"{self.eval_running_dt_start} ~ {self.eval_running_dt_end}",
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
        y_true = data[f'abort_{abortion_period}'].values
        y_pred = data[f'abort_{abortion_period}_decision'].values
        y_prob = data[f'abort_{abortion_period}_pred'].values

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
            result_row = {
                'stats_dt': f"{self.eval_running_dt_start} ~ {self.eval_running_dt_end}",
                f'l{level}_name': name,
                'total_sample_num': total_samples_num,
                'remain_sample_num': filtered_samples_num,
                'eval_period': abortion_period,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'auc': auc,
                'recognition': self.recognition
            }
        elif exclude_feiwen:
            result_row = {
                'stats_dt': f"{self.eval_running_dt_start} ~ {self.eval_running_dt_end}",
                'total_sample_num': total_samples_num,
                'remain_sample_num': filtered_samples_num,
                'eval_period': abortion_period,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'auc': auc,
                'recognition': self.recognition
            }
        else:
            result_row = {
                'stats_dt': f"{self.eval_running_dt_start} ~ {self.eval_running_dt_end}",
                'total_sample_num': total_samples_num,
                'eval_period': abortion_period,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'auc': auc,
                'recognition': self.recognition
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

        return self.result
    
    # 计算分级组织指标
    def eval_organizational_hierarchy(self):
        l2_results, l3_results, l4_results = self.eval_with_organizational_hierarchy()
        return l2_results, l3_results, l4_results