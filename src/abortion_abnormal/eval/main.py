import pandas as pd
import logging
import os.path
import sys
import warnings
from tqdm import tqdm
import utils.serialize as serialize
from abortion_abnormal.eval.eval_base import EvalBaseMixin
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, classification_report
import numpy as np

base_dir = os.path.split(os.path.realpath(__file__))[0]
program = os.path.basename(sys.argv[0])
logger = logging.getLogger(program)
logging.basicConfig(format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s: %(message)s',
                    datefmt='%a, %d %b %Y %H:%M:%S')
logging.root.setLevel(level=logging.INFO)
warnings.filterwarnings("ignore")

# 全集团测试
class AbortionAbnormalAllOnsetEval(EvalBaseMixin):
    def __init__(self, logger=None):
        self.index_sample = pd.DataFrame()
        self.index_ground_truth = pd.DataFrame()
        self.outcome_dt_end = None
        self.logger = logger

    # 构建评测数据集
    def build_eval_set(self, eval_running_dt_end, eval_interval, param=None):
        self.eval_running_dt_end = eval_running_dt_end
        eval_running_dt_end = pd.to_datetime(eval_running_dt_end)
        # 获取真实标签
        self.index_ground_truth = pd.read_csv("data/interim_data/abortion_calculate_data_label.csv", encoding='utf-8')
        # self.index_ground_truth = pd.read_csv("data/abortion_calculate_data_label.csv", encoding='utf-8')
        self.index_ground_truth['stats_dt'] = pd.to_datetime(self.index_ground_truth['stats_dt'])
        # 获取预测样本
        self.index_sample = pd.read_csv("data/predict/has_risk_predict_result.csv", encoding='utf-8')
        # self.index_sample = pd.read_csv("data/has_risk_predict_result.csv", encoding='utf-8')
        self.index_sample['stats_dt'] = pd.to_datetime(self.index_sample['stats_dt'])
        # 筛选日期范围内的样本
        self.index_ground_truth = self.index_ground_truth[
            (self.index_ground_truth['stats_dt'] >= eval_running_dt_end - pd.Timedelta(days=eval_interval)) &
            (self.index_ground_truth['stats_dt'] <= eval_running_dt_end)
        ]
        # 只保留stats_dt, pigfarm_dk, abortion_1_7,abortion_8_14,abortion_15_21列
        self.index_ground_truth = self.index_ground_truth[['stats_dt', 'pigfarm_dk', 'abortion_1_7', 'abortion_8_14', 'abortion_15_21']]
        # 筛选日期范围内的样本
        self.index_sample = self.index_sample[
            (self.index_sample['stats_dt'] >= eval_running_dt_end - pd.Timedelta(days=eval_interval)) &
            (self.index_sample['stats_dt'] <= eval_running_dt_end)
        ]
        # 只保留stats_dt, pigfarm_dk, prob_class_0, prob_class_1, prob_class_2, prob_class_3列
        self.index_sample = self.index_sample[['stats_dt', 'pigfarm_dk', 'prob_class_0', 'prob_class_1', 'prob_class_2', 'prob_class_3']]
        
        # 添加预测列 pred_1_7, pred_8_14, pred_15_21
        # 初始化为全0
        self.index_sample['pred_1_7'] = 0
        self.index_sample['pred_8_14'] = 0
        self.index_sample['pred_15_21'] = 0

        # 当prob_class_0 <= 0.5时，根据特定逻辑设置预测值
        mask = self.index_sample['prob_class_0'] <= 0.5
        if mask.any():
            # 获取prob_class_0 <= 0.5的行
            subset = self.index_sample[mask]
            
            # 计算每行的prob_class_1, prob_class_2, prob_class_3的平均值
            subset_avg = subset[['prob_class_1', 'prob_class_2', 'prob_class_3']].mean(axis=1)

            # 当prob_class值大于该行的平均值时，将对应的pred设为1
            self.index_sample.loc[mask, 'pred_1_7'] = (subset['prob_class_1'] > subset_avg).astype(int)
            self.index_sample.loc[mask, 'pred_8_14'] = (subset['prob_class_2'] > subset_avg).astype(int)
            self.index_sample.loc[mask, 'pred_15_21'] = (subset['prob_class_3'] > subset_avg).astype(int)

    def get_eval_index_sample(self):
        return self.index_sample
    
    def eval_increasing_risk_samples(self, merged_df, periods, is_1_2=True):
        # 筛选特定风险转变样本
        if( is_1_2 ):
            increasing_risk_idx = ((merged_df['abortion_1_7'] == 0) & (merged_df['abortion_8_14'] == 1))
            range_name = "1-7天到8-14天"
            pred_t = 'pred_1_7'
            pred_t_plus_1 = 'pred_8_14'
            truth_t = 'abortion_1_7'
            truth_t_plus_1 = 'abortion_8_14'
        else:
            increasing_risk_idx = ((merged_df['abortion_8_14'] == 0) & (merged_df['abortion_15_21'] == 1))
            range_name = "8-14天到15-21天"
            pred_t = 'pred_8_14'
            pred_t_plus_1 = 'pred_15_21'
            truth_t = 'abortion_8_14'
            truth_t_plus_1 = 'abortion_15_21'

        # 检查是否有符合条件的样本
        if increasing_risk_idx.sum() == 0:
            logger.info("\n未找到满足条件的风险转变样本")
            return {}
            
        # 获取符合条件的样本
        increasing_samples = merged_df[increasing_risk_idx]
        total_samples = len(increasing_samples)
        
        # logger.info(f"\n----- {range_name}风险上升样本评估 -----")
        # logger.info(f"符合条件的样本数量: {total_samples}")
        
        # 计算预测正确的样本数量
        correct_t = (increasing_samples[pred_t] == increasing_samples[truth_t]).sum()
        correct_t_plus_1 = (increasing_samples[pred_t_plus_1] == increasing_samples[truth_t_plus_1]).sum()
        
        # 计算完全正确的样本数量（两个时期都预测正确）
        both_correct = ((increasing_samples[pred_t] == increasing_samples[truth_t]) & 
                    (increasing_samples[pred_t_plus_1] == increasing_samples[truth_t_plus_1])).sum()
        
        # 计算模型预测为T和T+1都为1的样本数量
        both_pred_positive = ((increasing_samples[pred_t] == 1) & (increasing_samples[pred_t_plus_1] == 1)).sum()
        
        # logger.info(f"时期T预测正确的样本数: {correct_t} (占比: {correct_t/total_samples:.2%})")
        # logger.info(f"时期T+1预测正确的样本数: {correct_t_plus_1} (占比: {correct_t_plus_1/total_samples:.2%})")
        # logger.info(f"两个时期都预测正确的样本数: {both_correct} (占比: {both_correct/total_samples:.2%})")
        # logger.info(f"模型预测T和T+1都为1的样本数: {both_pred_positive} (占比: {both_pred_positive/total_samples:.2%})")
        
        # 保存结果
        increasing_results = {}
        increasing_results[f"increasing_{range_name}"] = {
            "count": total_samples,
            "correct_t": int(correct_t),
            "correct_t_plus_1": int(correct_t_plus_1),
            "both_correct": int(both_correct),
            "both_pred_positive": int(both_pred_positive)
        }
        
        return increasing_results
        
        
    def eval_decreasing_risk_samples(self, merged_df, periods, is_1_2=True):
        # 筛选特定风险下降样本
        if(is_1_2):
            decreasing_risk_idx = ((merged_df['abortion_1_7'] == 1) & (merged_df['abortion_8_14'] == 0))
            range_name = "1-7天到8-14天"
            pred_t = 'pred_1_7'
            pred_t_plus_1 = 'pred_8_14'
            truth_t = 'abortion_1_7'
            truth_t_plus_1 = 'abortion_8_14'
        else:
            decreasing_risk_idx = ((merged_df['abortion_8_14'] == 1) & (merged_df['abortion_15_21'] == 0))
            range_name = "8-14天到15-21天"
            pred_t = 'pred_8_14'
            pred_t_plus_1 = 'pred_15_21'
            truth_t = 'abortion_8_14'
            truth_t_plus_1 = 'abortion_15_21'

        # 检查是否有符合条件的样本
        if decreasing_risk_idx.sum() == 0:
            logger.info(f"\n未找到满足条件的风险下降样本 (从高到低): {range_name}")
            return {}
        
        # 获取符合条件的样本
        decreasing_samples = merged_df[decreasing_risk_idx]
        total_samples = len(decreasing_samples)

        # logger.info(f"\n----- {range_name}风险下降样本评估 -----")
        # logger.info(f"符合条件的样本数量: {total_samples}")
        
        # 计算预测正确的样本数量
        correct_t = (decreasing_samples[pred_t] == decreasing_samples[truth_t]).sum()
        correct_t_plus_1 = (decreasing_samples[pred_t_plus_1] == decreasing_samples[truth_t_plus_1]).sum()
        
        # 计算完全正确的样本数量（两个时期都预测正确）
        both_correct = ((decreasing_samples[pred_t] == decreasing_samples[truth_t]) & 
                       (decreasing_samples[pred_t_plus_1] == decreasing_samples[truth_t_plus_1])).sum()
        
        # 计算模型预测为T和T+1都为0的样本数量
        both_pred_negative = ((decreasing_samples[pred_t] == 1) & (decreasing_samples[pred_t_plus_1] == 1)).sum()
        
        # logger.info(f"时期T预测正确的样本数: {correct_t} (占比: {correct_t/total_samples:.2%})")
        # logger.info(f"时期T+1预测正确的样本数: {correct_t_plus_1} (占比: {correct_t_plus_1/total_samples:.2%})")
        # logger.info(f"两个时期都预测正确的样本数: {both_correct} (占比: {both_correct/total_samples:.2%})")
        # logger.info(f"模型预测T和T+1都为1的样本数: {both_pred_negative} (占比: {both_pred_negative/total_samples:.2%})")
        
        # 保存结果
        decreasing_results = {}
        decreasing_results[f"decreasing_{range_name}"] = {
            "count": total_samples,
            "correct_t": int(correct_t),
            "correct_t_plus_1": int(correct_t_plus_1),
            "both_correct": int(both_correct),
            "both_pred_negative": int(both_pred_negative)
        }
        
        return decreasing_results
        

    def save_results_to_csv(self, results, pig_farm_range, output_path="data/eval_results/" ):
        # 创建输出目录（如果不存在）
        os.makedirs(output_path, exist_ok=True)
        
        # 获取指标值
        metrics_1_7 = results.get("1-7 days", {})
        metrics_8_14 = results.get("8-14 days", {})
        metrics_15_21 = results.get("15-21 days", {})
                
        # 准备数据行
        rows = []
        
        # 评估指标列表
        metrics = ['precision', 'recall', 'f1_score', 'auc']
        
        for metric in metrics:
            # 获取样本数量
            sample_num = metrics_1_7.get("sample_num", 0)

            # 获取每个时间段的指标值
            val_1_7 = metrics_1_7.get(metric, float('nan'))
            val_8_14 = metrics_8_14.get(metric, float('nan'))
            val_15_21 = metrics_15_21.get(metric, float('nan'))
            
            # 计算平均值
            val_1_21 = np.nanmean([val_1_7, val_8_14, val_15_21])
            
            # 添加行
            rows.append({
                'stats_dt': self.eval_running_dt_end,
                'sample_num': sample_num,
                'eval': metric,
                '1_21': val_1_21,
                '1_7': val_1_7,
                '8_14': val_8_14, 
                '15_21': val_15_21
            })
        
        # 创建DataFrame
        df = pd.DataFrame(rows)
        
        # 设置文件名
        filename = f"overall_metrics_{self.eval_running_dt_end}_{pig_farm_range}.csv"
        file_path = os.path.join(output_path, filename)
        
        # 保存CSV
        df.to_csv(file_path, index=False, encoding='utf-8')
        
        return file_path


    def save_special_results_to_csv(self, results, pig_farm_range, output_path="data/eval_results/"):
        # 创建输出目录（如果不存在）
        os.makedirs(output_path, exist_ok=True)
        
        # 准备数据行
        rows = []
        
        # 映射结果键到评估名称
        eval_mapping = {
            "decreasing_1-7天到8-14天": "above_down_1_14",
            "decreasing_8-14天到15-21天": "above_down_8_21",
            "increasing_1-7天到8-14天": "down_above_1_14",
            "increasing_8-14天到15-21天": "down_above_8_21"
        }
        
        # 处理每种特殊情况
        for result_key, eval_name in eval_mapping.items():
            if result_key in results:
                metrics = results[result_key]
                total = metrics.get("count", 0)
                
                if total > 0:
                    # 计算各项指标占比
                    prev_precision = metrics.get("correct_t", 0) / total if total else 0
                    next_precision = metrics.get("correct_t_plus_1", 0) / total if total else 0
                    both_precision = metrics.get("both_correct", 0) / total if total else 0
                    
                    # 对于上升和下降风险选择正确的字段
                    if "increasing" in result_key:
                        both_1 = metrics.get("both_pred_positive", 0) / total if total else 0
                    else:
                        both_1 = metrics.get("both_pred_negative", 0) / total if total else 0
                    
                    # 添加行
                    rows.append({
                        'stats_dt': self.eval_running_dt_end,
                        'sample_num': total,  # 使用特定情况的样本数量
                        'eval': eval_name,
                        'prev_precision': prev_precision,
                        'next_precision': next_precision,
                        'both_precision': both_precision,
                        'both_1': both_1
                    })
        
        # 如果有结果，创建并保存DataFrame
        if rows:
            df = pd.DataFrame(rows)
            
            # 设置文件名
            filename = f"special_metrics_{self.eval_running_dt_end}_{pig_farm_range}.csv"
            file_path = os.path.join(output_path, filename)
            
            # 保存CSV
            df.to_csv(file_path, index=False, encoding='utf-8')
            logger.info(f"特殊样本评估结果已保存到: {file_path}")
            return file_path
        else:
            logger.warning("没有特殊样本评估结果可供保存")
            return None


    def eval_with_index_sample(self, pig_farm_range="整体"):
        # 根据stats_dt和pigfarm_dk将真实值和预测值进行合并
        merged_df = pd.merge(
            self.index_ground_truth,
            self.index_sample,
            on=['stats_dt', 'pigfarm_dk'],
            how='inner'
        )

        # 根据猪场范围进行数据筛选
        if pig_farm_range != "整体":
            # 加载猪场部门映射数据
            org_inv_map = pd.read_csv("data/raw_data/dim_org_inv.csv", encoding='utf-8')
            
            # 创建猪场编码到部门的映射
            farm_to_dept = dict(zip(org_inv_map['org_inv_dk'], org_inv_map['l2_org_inv_nm']))
            
            # 为merged_df添加部门列
            merged_df['department'] = merged_df['pigfarm_dk'].map(farm_to_dept)
            
            # 根据部门进行筛选
            if pig_farm_range == "猪业一部":
                merged_df = merged_df[merged_df['department'] == "猪业一部"]
                logger.info(f"筛选猪业一部数据，保留{len(merged_df)}行")
            elif pig_farm_range == "猪业二部":
                merged_df = merged_df[merged_df['department'] == "猪业二部"]
                logger.info(f"筛选猪业二部数据，保留{len(merged_df)}行")
            elif pig_farm_range == "猪业三部":
                merged_df = merged_df[merged_df['department'] == "猪业三部"]
                logger.info(f"筛选猪业三部数据，保留{len(merged_df)}行")

        # 用于调试
        # merged_df.to_csv('data/merged_df.csv', index=False)

        if merged_df.empty:
            logger.warning("合并后的数据集为空，请检查数据源和合并条件。")
            return None
        
        # 定义时间段和对应的列名
        periods = [
            ("1-7 days", 'abortion_1_7', 'pred_1_7'),
            ("8-14 days", 'abortion_8_14', 'pred_8_14'),
            ("15-21 days", 'abortion_15_21', 'pred_15_21')
        ]
        
        results = {}
        
        # 对于每个时间段，计算评估指标
        for period_name, truth_col, pred_col in periods:
            # 直接使用预先计算的二元预测值
            pred_binary = merged_df[pred_col]
            true_labels = merged_df[truth_col]
            
            # 计算指标
            precision = precision_score(true_labels, pred_binary, zero_division=0)
            recall = recall_score(true_labels, pred_binary, zero_division=0)
            f1 = f1_score(true_labels, pred_binary, zero_division=0)
            
            # 对于AUC计算，我们需要概率值而非二元值
            # 根据您的数据结构选择相应的概率列
            prob_col = None
            if pred_col == 'pred_1_7':
                prob_col = 'prob_class_1'
            elif pred_col == 'pred_8_14':
                prob_col = 'prob_class_2'
            elif pred_col == 'pred_15_21':
                prob_col = 'prob_class_3'
            
            # 计算AUC（避免只有一类的情况）
            try:
                auc = roc_auc_score(true_labels, merged_df[prob_col])
            except:
                auc = float('nan')
            
            # 保存结果
            results[period_name] = {
                "sample_num": len(true_labels),
                "precision": precision,
                "recall": recall,
                "f1_score": f1,
                "auc": auc
            }

            # logger.info(f"\n整体样本 - {period_name}:")
            # logger.info(f"样本数量: {len(true_labels)}")
            # logger.info(f"精确率: {precision:.4f}")
            # logger.info(f"召回率: {recall:.4f}")
            # logger.info(f"F1 Score: {f1:.4f}")
            # logger.info(f"AUC: {auc:.4f}")        
        
        # 计算风险变化样本的指标 (低到高)
        increasing_risk_results = self.eval_increasing_risk_samples(merged_df, periods, is_1_2=True)
        decreasing_risk_results = self.eval_decreasing_risk_samples(merged_df, periods, is_1_2=True)

        # 将各种风险样本的结果添加到总结果中
        if increasing_risk_results:
            results.update(increasing_risk_results)
        if decreasing_risk_results:
            results.update(decreasing_risk_results)

        # 计算风险下降样本的指标 (高到低)
        increasing_risk_results = self.eval_increasing_risk_samples(merged_df, periods, is_1_2=False)
        decreasing_risk_results = self.eval_decreasing_risk_samples(merged_df, periods, is_1_2=False)
        
        # 将各种风险样本的结果添加到总结果中
        if increasing_risk_results:
            results.update(increasing_risk_results)
        if decreasing_risk_results:
            results.update(decreasing_risk_results)
        
        # 将结果保存为CSV文件
        csv_path = self.save_results_to_csv(results, pig_farm_range=pig_farm_range)
        logger.info(f"评测结果保存到 {csv_path}")

        # 将特殊样本结果保存为CSV文件
        special_csv_path = self.save_special_results_to_csv(results, pig_farm_range=pig_farm_range)
        if special_csv_path:
            logger.info(f"特殊样本评测结果保存到 {special_csv_path}")

        return results

# 猪业一部测试
class AbortionAbnormalFirstOnsetEval(EvalBaseMixin):
    pass

# 猪业二部测试
class AbortionAbnormalSecondOnsetEval(EvalBaseMixin):
    pass

# 猪业三部测试
class AbortionAbnormalThirdOnsetEval(EvalBaseMixin):
    pass


if __name__ == "__main__":
    abortion_abnormal_eval = AbortionAbnormalAllOnsetEval()