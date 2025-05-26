import sys
sys.path.append("..")
import pandas as pd
import configs.feature_config as config
# import config.alert_base_config as config
import utils.serialize as serialize

class DataPreProcessMain(object):

    def __init__(self, **param):

        self.pig_farm_production_data = None
        self.pig_farm_intro_data = None
        self.logger = param.get('logger')
        self.running_date = param.get('running_date')
        self.running_date=pd.to_datetime(self.running_date)


    def get_dataset(self):
        # ====================================================================================读取原始数据 ====================================================================================
       
        self.logger.info("读取预处理前的文件，原始文件 or 中间文件")
        # 读取生产表数据
        self.pig_farm_production_data = serialize.datafile_read(config.DataPathConfig.ML_DATA_PATH.value)
        # 读取引种数据
        self.pig_farm_intro_data = serialize.datafile_read(config.DataPathConfig.W01_AST_BOAR_PATH.value)

     # ====================================================================================读取原始数据 ====================================================================================
      
    # 检查生产数据
    def check_production_info_data(self):
        # 检查是否存在缺失列
        feature_name_list=["STATS_DT", " PIGFARM_DK", " BASESOW_SQTY", " BASEMPIG_SQTY", " RESERVE_SOW_SQTY", " RESERVE_MPIG_SQTY", " SUCKHOUSE_BOAR_SQTY", " CAREHOUSE_BOAR_SQTY", " GROWHOUSE_BOAR_SQTY", " BASESOW_CULL_QTY", " BASESOW_DEATH_QTY", " BASESOW_ELIMINATE_QTY", " RESERVE_SOW_CULL_QTY", " RESERVE_SOW_DEATH_QTY", " RESERVE_SOW_ELIMINATE_QTY", " BOAR_CULL_QTY", " BOAR_DEATH_QTY", " BOAR_ELIMINATE_QTY", " BOAR_TRANSOUT_QTY", " BOAR_TRANSIN_QTY", " BOAR_SALE_QTY", " MATING_QTY", " ABORT_QTY", " NONPREG_QTY", " RELOVE_QTY", " ELIMINATE_QTY", " DEATH_QTY", " BEFORE114_MATING_FQTY", " BEFORE114_CBIRTH_FQTY", " BEFORE114_ESTIMATE_CBIRTH_FQTY", " CBIRTH_BROOD_QTY", " CBIRTH_TOTAL_PIGLET_QTY", " CBIRTH_LPIGLET_QTY", " CBIRTH_HPIGLET_QTY", " DFMPIGLET_REMAIN_QTY", " CBIRTH_PIGLET_BROOD_WT", " CBIRTH_PIGLET_AVG_WT", " WEAN_BROOD_QTY", " WEAN_HPIGLET_QTY", " PIGLET_CULL_QTY", " CBIRTH_PIGLET_CULL_QTY", " CARE_PIGLET_CULL_QTY", " RAISING_PIGLET_CULL_QTY", " PREG_STOCK_QTY"]
        columns_to_check_set = set(feature_name_list)
        missing_columns = columns_to_check_set - set(self.pig_farm_production_data.columns)

        if len(missing_columns)>0:
            self.logger.error('生产特征数据存在缺失列: %s', missing_columns)

        # 检查列org_id是否存在空值
        org_id_has_nulls = self.pig_farm_production_data['pigfarm_dk'].isna().any()
        if org_id_has_nulls:
            self.logger.error('生产特征数据org_id存在缺失值!')

        # 检查列统计日期stats_dt是否存在空值 
        date_code_has_nulls = self.pig_farm_production_data['stats_dt'].isna().any()
        if date_code_has_nulls:
            self.logger.error('生产特征数据date_code存在缺失值!')

        # 将列stats_dt转换为datetime格式
        self.pig_farm_production_data["stats_dt"] = pd.to_datetime(self.pig_farm_production_data["stats_dt"])
        
        # 检查列stats_dt是否存在大于当前时间的数据
        check_production_time_range=self.pig_farm_production_data[self.pig_farm_production_data["stats_dt"]>=self.running_date]
        
        if len(check_production_time_range)>0:
            self.logger.error('生产特征数据存在大于运行时间的数据,注意时间边界!')

        self.logger.info('----------生产特征数据检查完毕并上报----------')

    # 检查引种数据
    def check_intro_info_data(self):
        # 检查是否存在缺失列
        feature_name_list=["INTRO_DT", "org_inv_nm", "org_inv_dk", "EAR_NO", "BOAR_SRC_TYPE", "VENDOR_NM", "CFFROMHOGP_NM"]
        columns_to_check_set = set(feature_name_list)
        missing_columns = columns_to_check_set - set(self.pig_farm_intro_data.columns)

        if len(missing_columns)>0:
            self.logger.error('生产特征数据存在缺失列: %s', missing_columns)

        # 检查列org_id是否存在空值
        org_id_has_nulls = self.pig_farm_intro_data['pigfarm_dk'].isna().any()
        if org_id_has_nulls:
            self.logger.error('生产特征数据org_id存在缺失值!')

        # 检查列统计日期stats_dt是否存在空值 
        date_code_has_nulls = self.pig_farm_intro_data['stats_dt'].isna().any()
        if date_code_has_nulls:
            self.logger.error('生产特征数据date_code存在缺失值!')

        # 将列stats_dt转换为datetime格式
        self.pig_farm_intro_data["stats_dt"] = pd.to_datetime(self.pig_farm_intro_data["stats_dt"])
        
        # 检查列stats_dt是否存在大于当前时间的数据
        check_intro_time_range=self.pig_farm_intro_data[self.pig_farm_intro_data["stats_dt"]>=self.running_date]
        
        if len(check_intro_time_range)>0:
            self.logger.error('引种特征数据存在大于运行时间的数据,注意时间边界!')

        self.logger.info('----------引种特征数据检查完毕并上报----------')

    def check_data(self):
        self.logger.info('----------开始检查数据----------')
        self.get_dataset()
        self.logger.info('----------检查生产表----------')
        self.check_production_info_data()
        self.logger.info('----------检查引种表----------')
        self.check_intro_info_data()
        self.logger.info('----------数据检测完毕!----------')
