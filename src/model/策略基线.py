import pandas as pd

def prrs_filter(data):
            # 过滤掉不需要的检查项
        """
        筛选PRRS数据
        :param check_data: 输入数据
        :return: 筛选后的数据
        """
        # 定义特殊项目ID列表
        special_items = [
            'bDoAA065yHyCrSt1',
            'bDoAAzoDnCKCrSt1',
            'bDoAAvklLFWCrSt1',
        ]
        
        # 定义所有需要的check_item_dk
        all_items = [
            # 野毒
            'bDoAAfRM6YiCrSt1',
            'bDoAArPPgj6CrSt1',
            'bDoAAfRM6IGCrSt1',
            'bDoAAfYsNUGCrSt1',
            'bDoAAfYsM8eCrSt1',
            'bDoAAfYr79SCrSt1',
            # 抗原
            'bDoAAJyZSTSCrSt1',
            'bDoAAfYgkW2CrSt1',
            'bDoAAfYq6LWCrSt1',
            'bDoAAfYq6kWCrSt1',
            'bDoAAfYsNKyCrSt1',
            'bDoAAwWyhPOCrSt1',
            # 抗体
            'bDoAAJyZSZiCrSt1',
            # 特殊项目
            'bDoAA065yHyCrSt1',
            'bDoAAzoDnCKCrSt1',
            'bDoAAvklLFWCrSt1',
        ]
        
        # 定义需要的index_item_dk
        valid_indexes = [
            # 野毒
            'bDoAAfYcdbLWD/D5',
            'bDoAAfYcdbTWD/D5',
            'bDoAAfRPf0jWD/D5',
            'bDoAAKqewlzWD/D5',
            # 条带
            'bDoAAKqffmXWD/D5',
            'bDoAAKqewhjWD/D5',
            # prrsvct
            'bDoAAKqffxXWD/D5',
            # 抗原
            'bDoAAfYq6kvWD/D5',
            # 抗体
            'bDoAAKqZiKzWD/D5',
            # s/p
            'bDoAAKqZiKzWD/D5',
        ]
        
        # 步骤1: 首先按照check_item_dk筛选所有数据
        filtered_data = data[data['check_item_dk'].isin(all_items)]
        
        # 步骤2: 将数据分为两部分
        # 非特殊项目数据 - 直接保留
        regular_items = filtered_data[~filtered_data['check_item_dk'].isin(special_items)]
        
        # 特殊项目数据 - 需要进一步筛选index_item_dk
        special_items_data = filtered_data[filtered_data['check_item_dk'].isin(special_items)]
        filtered_special_items = special_items_data[special_items_data['index_item_dk'].isin(valid_indexes)]
        
        # 步骤3: 合并两部分数据
        final_data = pd.concat([regular_items, filtered_special_items])
        
        return final_data

def calculate_piglet_risk(row, data):
    stats = row['stats_dt']
    pigfarm_dk = row['pigfarm_dk']

    # 过滤数据
    filtered_data = data[
        (data['stats_dt'] == stats) & (data['pigfarm_dk'] == pigfarm_dk)
    ]
    if filtered_data.empty:
        return False

def calculate_tame_risk(row, check_data, tame_data):
    stats = row['stats_dt']
    pigfarm_dk = row['pigfarm_dk']

    


if __name__ == "__main__":
    piglet_data = pd.read_csv(r'data\raw_data\ads_pig_efficient_piglet_batch_analysis_day.csv')
    tame_data = pd.read_csv(r'data\raw_data\TMP_ADS_PIG_ISOLATION_TAME_RISK_L1_N2.csv')
    intro_data = pd.read_csv(r'data\raw_data\W01_AST_BOAR.csv')
    production_data = pd.read_csv(r'data\raw_data\ads_pig_org_total_to_ml_training_day  生产数据.csv')
    check_data = pd.read_csv(r'data\raw_data\TMP_PIG_ORG_DISEASE_CHECK_RESULT_DAY  检测数据猪场.csv')
    check_data = prrs_filter(check_data, production_data)

    index_start = pd.to_datetime('2024-12-01')
    index_end = pd.to_datetime('2025-01-30')

    # 过滤数据
    filtered_production = production_data[
        (production_data['stats_dt'] >= index_start) & (production_data['stats_dt'] <= index_end)
    ]

    grouped_data = filtered_production.groupby('pigfarm_dk')
    for pigfarm_dk, group in grouped_data:

        for index, row in group.iterrows():
            piglet_bool = calculate_piglet_risk(row, piglet_data)