import pandas as pd
import logging
import os
import sys
from abortion_abnormal.eval.eval_base import EvalBaseMixin
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, classification_report

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
        org_mapping = dim_org_inv[['org_inv_dk', 'l2_org_inv_nm']].copy()
        
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

        # 分别为三个部门创建数据子集
        self.division_data = {
            '整体': self.predict_data,  # 保留整体数据
            '猪业一部': self.predict_data[self.predict_data['l2_org_inv_nm'] == '猪业一部'],
            '猪业二部': self.predict_data[self.predict_data['l2_org_inv_nm'] == '猪业二部'],
            '猪业三部': self.predict_data[self.predict_data['l2_org_inv_nm'] == '猪业三部']
        }


    def get_eval_index_sample(self):
        return self.index_sample


    # 计算前一天为prev_status，后一天为next_status的指标
    def eval_one_periods_metric(self, period, prev_status, next_status, organization):
        # 1. 筛选特殊样本
        special_samples = []
        
        # 按照猪场分组
        grouped = self.division_data[organization].groupby('pigfarm_dk')
        
        for pigfarm, group_data in grouped:
            # 按时间排序
            sorted_data = group_data.sort_values('stats_dt')

            # 找出时间t时abortion_period为0，时间t+1时abortion_period为1的时间点
            for i in range(len(sorted_data) - 1):
                if sorted_data.iloc[i][f'abortion_{period}'] == prev_status and sorted_data.iloc[i+1][f'abortion_{period}'] == next_status:
                    t = sorted_data.iloc[i]['stats_dt']
                    
                    # 找出t-3到t+4的所有样本
                    farm_data = sorted_data.copy()
                    t_minus_3 = t - pd.Timedelta(days=3)
                    t_plus_4 = t + pd.Timedelta(days=4)
                    
                    # 筛选时间范围内的样本
                    time_range_samples = farm_data[(farm_data['stats_dt'] >= t_minus_3) & 
                                                  (farm_data['stats_dt'] <= t_plus_4)]
                    
                    # 添加到特殊样本列表
                    special_samples.append(time_range_samples)
        
        # 合并所有特殊样本
        if len(special_samples) > 0:
            self.index_sample = pd.concat(special_samples)
            # 2. 去重
            self.index_sample = self.index_sample.drop_duplicates()
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
            auc = roc_auc_score(y_true, y_prob)
        except:
            # 处理AUC计算异常情况（例如只有一个类别）
            if self.logger:
                self.logger.warning("无法计算AUC，可能是样本中只存在一个类别")
        
        # 保存结果
        result_row = {
            'description': f'从{prev_status}到{next_status}',
            'stats_dt': self.eval_running_dt_end,
            'sample_num': len(self.index_sample),
            'eval': period,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'auc': auc
        }
        self.result = pd.concat([self.result, pd.DataFrame([result_row])], ignore_index=True)
        
        # 添加详细的分类报告
        # report = classification_report(y_true, y_pred, output_dict=True)
        
    
    def eval_with_index_sample(self):
        results = []
        for organization in ['整体', '猪业一部', '猪业二部', '猪业三部']:
            # 分别计算t未超过0.25%且t+1超过0.25%、t超过0.25%且t+1未超过0.25%、t和t+1都超过0.25%的指标
            for prev_status, next_status in [(0, 1), (1, 0), (1, 1)]:
                for abortion_period in ['1_7', '8_14', '15_21']:
                    # 评估从低到高的指标
                    self.eval_one_periods_metric(abortion_period, prev_status, next_status, organization)
            # 保存结果
            results.append(self.result)
            self.result = pd.DataFrame()  # 清空结果以便下次使用
        
        # 保存到文件
        with pd.ExcelWriter(f"data/eval/abortion_abnormal_eval_{self.eval_running_dt_end}.xlsx") as writer:
            results[0].to_excel(writer, sheet_name='整体', index=False)
            results[1].to_excel(writer, sheet_name='猪业一部', index=False)
            results[2].to_excel(writer, sheet_name='猪业二部', index=False)
            results[3].to_excel(writer, sheet_name='猪业三部', index=False)

        print(self.result)


# 任务2评估
class AbortionAbnormalEval2(EvalBaseMixin):
    pass

# 任务3评估
class AbortionAbnormalEval3(EvalBaseMixin):
    pass