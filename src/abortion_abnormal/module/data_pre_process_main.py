import sys
sys.path.append("..")
import os
import pandas as pd
import configs.feature_config as config
# import config.alert_base_config as config
import utils.serialize as serialize

class DataPreProcessMain(object):

    def __init__(self, **param):

        self.pig_farm_production_data = None
        self.pig_farm_intro_data = None
        self.logger = param.get('logger')
        self.predict_running_dt = param.get('predict_running_dt')
        self.predict_interval = param.get('predict_interval')

        self.predict_running_start_dt = pd.to_datetime(self.predict_running_dt) + pd.Timedelta(days=1)
        self.predict_running_end_dt = pd.to_datetime(self.predict_running_dt) + pd.Timedelta(days=self.predict_interval)


    def get_dataset(self):
        # ====================================================================================读取原始数据 ====================================================================================

        self.logger.info("读取预处理前的文件，原始文件 or 中间文件")
        # 读取生产表数据
        self.pig_farm_production_data = pd.read_csv('data/raw_data/ads_pig_org_total_to_ml_training_day  生产数据.csv', low_memory=False)
        # 读取引种数据
        self.pig_farm_intro_data = pd.read_csv('data/raw_data/W01_AST_BOAR.csv', low_memory=False)
        # 读取入群数据
        self.pig_farm_tame_data = pd.read_csv('data/raw_data/TMP_ADS_PIG_ISOLATION_TAME_RISK_L1_N2.csv', low_memory=False)
        # 读取检测数据
        self.pig_farm_check_data = pd.read_csv('data/raw_data/TMP_PIG_ORG_DISEASE_CHECK_RESULT_DAY  检测数据猪场.csv', low_memory=False)
        # 读取死淘数据
        self.pig_farm_death_data = pd.read_csv('data/raw_data/TMP_ORG_PRRS_OVERALL_ADOPT_CULL_DAY.csv', low_memory=False)

        # ML_DATA_PATH = RAW_DATA_DIR / "ads_pig_org_total_to_ml_training_day  生产数据.csv"
        # W01_AST_BOAR_PATH = RAW_DATA_DIR / "W01_AST_BOAR.csv"
        # TMP_ADS_PIG_ISOLATION_TAME_RISK_L1_N2 = RAW_DATA_DIR / "TMP_ADS_PIG_ISOLATION_TAME_RISK_L1_N2.csv"
        # PRRS_CHECK_DATA_PATH = RAW_DATA_DIR / "TMP_PIG_ORG_DISEASE_CHECK_RESULT_DAY  检测数据猪场.csv"
        # DEATH_CONFIRM_DATA_PATH = RAW_DATA_DIR / "TMP_ORG_PRRS_OVERALL_ADOPT_CULL_DAY.csv"

     # ====================================================================================读取原始数据 ====================================================================================
      
    # 检查生产数据
    def check_production_info_data(self):
        # # 检查是否存在缺失列
        # feature_name_list=["STATS_DT", " PIGFARM_DK", " BASESOW_SQTY", " BASEMPIG_SQTY", " RESERVE_SOW_SQTY", " RESERVE_MPIG_SQTY", " SUCKHOUSE_BOAR_SQTY", " CAREHOUSE_BOAR_SQTY", " GROWHOUSE_BOAR_SQTY", " BASESOW_CULL_QTY", " BASESOW_DEATH_QTY", " BASESOW_ELIMINATE_QTY", " RESERVE_SOW_CULL_QTY", " RESERVE_SOW_DEATH_QTY", " RESERVE_SOW_ELIMINATE_QTY", " BOAR_CULL_QTY", " BOAR_DEATH_QTY", " BOAR_ELIMINATE_QTY", " BOAR_TRANSOUT_QTY", " BOAR_TRANSIN_QTY", " BOAR_SALE_QTY", " MATING_QTY", " ABORT_QTY", " NONPREG_QTY", " RELOVE_QTY", " ELIMINATE_QTY", " DEATH_QTY", " BEFORE114_MATING_FQTY", " BEFORE114_CBIRTH_FQTY", " BEFORE114_ESTIMATE_CBIRTH_FQTY", " CBIRTH_BROOD_QTY", " CBIRTH_TOTAL_PIGLET_QTY", " CBIRTH_LPIGLET_QTY", " CBIRTH_HPIGLET_QTY", " DFMPIGLET_REMAIN_QTY", " CBIRTH_PIGLET_BROOD_WT", " CBIRTH_PIGLET_AVG_WT", " WEAN_BROOD_QTY", " WEAN_HPIGLET_QTY", " PIGLET_CULL_QTY", " CBIRTH_PIGLET_CULL_QTY", " CARE_PIGLET_CULL_QTY", " RAISING_PIGLET_CULL_QTY", " PREG_STOCK_QTY"]
        # columns_to_check_set = set(feature_name_list)
        # missing_columns = columns_to_check_set - set(self.pig_farm_production_data.columns)

        # if len(missing_columns)>0:
        #     self.logger.error('生产特征数据存在缺失列: %s', missing_columns)

        # # 检查列org_id是否存在空值
        # org_id_has_nulls = self.pig_farm_production_data['pigfarm_dk'].isna().any()
        # if org_id_has_nulls:
        #     self.logger.error('生产特征数据org_id存在缺失值!')

        # # 检查列统计日期stats_dt是否存在空值 
        # date_code_has_nulls = self.pig_farm_production_data['stats_dt'].isna().any()
        # if date_code_has_nulls:
        #     self.logger.error('生产特征数据date_code存在缺失值!')

        # # 将列stats_dt转换为datetime格式
        # self.pig_farm_production_data["stats_dt"] = pd.to_datetime(self.pig_farm_production_data["stats_dt"])
        
        # # 检查列stats_dt是否存在大于当前时间的数据
        # check_production_time_range=self.pig_farm_production_data[self.pig_farm_production_data["stats_dt"]>=self.running_date]
        
        # if len(check_production_time_range)>0:
        #     self.logger.error('生产特征数据存在大于运行时间的数据,注意时间边界!')

        # self.logger.info('----------生产特征数据检查完毕并上报----------')
        pass

    # 检查引种数据
    def check_intro_info_data(self):
        # # 检查是否存在缺失列
        # feature_name_list=["INTRO_DT", "org_inv_nm", "org_inv_dk", "EAR_NO", "BOAR_SRC_TYPE", "VENDOR_NM", "CFFROMHOGP_NM"]
        # columns_to_check_set = set(feature_name_list)
        # missing_columns = columns_to_check_set - set(self.pig_farm_intro_data.columns)

        # if len(missing_columns)>0:
        #     self.logger.error('生产特征数据存在缺失列: %s', missing_columns)

        # # 检查列org_id是否存在空值
        # org_id_has_nulls = self.pig_farm_intro_data['pigfarm_dk'].isna().any()
        # if org_id_has_nulls:
        #     self.logger.error('生产特征数据org_id存在缺失值!')

        # # 检查列统计日期stats_dt是否存在空值 
        # date_code_has_nulls = self.pig_farm_intro_data['stats_dt'].isna().any()
        # if date_code_has_nulls:
        #     self.logger.error('生产特征数据date_code存在缺失值!')

        # # 将列stats_dt转换为datetime格式
        # self.pig_farm_intro_data["stats_dt"] = pd.to_datetime(self.pig_farm_intro_data["stats_dt"])
        
        # # 检查列stats_dt是否存在大于当前时间的数据
        # check_intro_time_range=self.pig_farm_intro_data[self.pig_farm_intro_data["stats_dt"]>=self.running_date]
        
        # if len(check_intro_time_range)>0:
        #     self.logger.error('引种特征数据存在大于运行时间的数据,注意时间边界!')

        # self.logger.info('----------引种特征数据检查完毕并上报----------')
        pass

    def check_data(self):
        self.logger.info('----------开始检查数据----------')
        self.get_dataset()
        self.logger.info('----------检查生产表----------')
        self.check_production_info_data()
        self.logger.info('----------检查引种表----------')
        self.check_intro_info_data()
        self.logger.info('----------数据检测完毕!----------')

    def create_eval_dataset(self):
        # todo 创建评估数据集目录
        self.eval_dataset_path = 'data/eval_train_data'

        os.makedirs(self.eval_dataset_path, exist_ok=True)


        # todo
        predict_running_start_dts = ['2024-03-01', '2024-06-01', '2024-09-01', '2024-12-01']
        predict_running_end_dts = ['2024-03-30', '2024-06-30', '2024-09-30', '2024-12-30']
        # 创建索引文件
        # for start_dt, end_dt in zip(predict_running_start_dts, predict_running_end_dts):
        #     tmp_data = self.pig_farm_production_data[['stats_dt', 'pigfarm_dk']].drop_duplicates()
        #     tmp_data['stats_dt'] = pd.to_datetime(tmp_data['stats_dt'])
        #     start_dt = pd.to_datetime(start_dt)
        #     end_dt = pd.to_datetime(end_dt)
        #     tmp_data = tmp_data[(tmp_data['stats_dt'] >= start_dt) & (tmp_data['stats_dt'] <= end_dt)]
        #     tmp_data.to_csv(os.path.join(self.eval_dataset_path, f'index_sample_{start_dt.strftime("%Y%m%d")}.csv'), index=False)


        for start_dt, end_dt in zip(predict_running_start_dts, predict_running_end_dts):
            # 创建训练集
            # 设置训练集最大时间
            start_dt = pd.to_datetime(start_dt)
            end_dt = pd.to_datetime(end_dt)
            max_train_dt = pd.to_datetime(start_dt) - pd.Timedelta(days=1)
            print(f"开始生成评估数据集：{start_dt.strftime('%Y-%m-%d')} - {end_dt.strftime('%Y-%m-%d')}")

            # 设置数据保存目录
            train_dataset_path = os.path.join(self.eval_dataset_path, f'{start_dt.strftime("%Y%m%d")}_{end_dt.strftime("%Y%m%d")}')
            os.makedirs(train_dataset_path, exist_ok=True)

            # todo 生成训练集
            self.logger.info('----------开始生成评估用训练数据集----------')
            # 过滤生产数据
            self.pig_farm_production_data['stats_dt'] = pd.to_datetime(self.pig_farm_production_data['stats_dt'])
            pig_farm_production_data = self.pig_farm_production_data[self.pig_farm_production_data['stats_dt'] <= max_train_dt]
            # 过滤引种数据
            self.pig_farm_intro_data['intro_dt'] = pd.to_datetime(self.pig_farm_intro_data['intro_dt'])
            pig_farm_intro_data = self.pig_farm_intro_data[self.pig_farm_intro_data['intro_dt'] <= max_train_dt]
            # 过滤入群数据
            self.pig_farm_tame_data['tmp_ads_pig_isolation_tame_risk_l1_n2.bill_dt'] = pd.to_datetime(self.pig_farm_tame_data['tmp_ads_pig_isolation_tame_risk_l1_n2.bill_dt'])
            pig_farm_tame_data = self.pig_farm_tame_data[self.pig_farm_tame_data['tmp_ads_pig_isolation_tame_risk_l1_n2.bill_dt'] <= max_train_dt]
            # 过滤检测数据
            self.pig_farm_check_data['receive_dt'] = pd.to_datetime(self.pig_farm_check_data['receive_dt'])
            pig_farm_check_data = self.pig_farm_check_data[self.pig_farm_check_data['receive_dt'] <= max_train_dt]
            # 过滤死淘数据
            self.pig_farm_death_data['stats_dt'] = pd.to_datetime(self.pig_farm_death_data['stats_dt'])
            pig_farm_death_data = self.pig_farm_death_data[self.pig_farm_death_data['stats_dt'] <= max_train_dt]
            # 保存训练集数据
            pig_farm_production_data.to_csv(os.path.join(train_dataset_path, 'ads_pig_org_total_to_ml_training_day  生产数据.csv'), index=False)
            pig_farm_intro_data.to_csv(os.path.join(train_dataset_path, 'W01_AST_BOAR.csv'), index=False)
            pig_farm_tame_data.to_csv(os.path.join(train_dataset_path, 'TMP_ADS_PIG_ISOLATION_TAME_RISK_L1_N2.csv'), index=False)
            pig_farm_check_data.to_csv(os.path.join(train_dataset_path, 'TMP_PIG_ORG_DISEASE_CHECK_RESULT_DAY  检测数据猪场.csv'), index=False)
            pig_farm_death_data.to_csv(os.path.join(train_dataset_path, 'TMP_ORG_PRRS_OVERALL_ADOPT_CULL_DAY.csv'), index=False)

        assert False