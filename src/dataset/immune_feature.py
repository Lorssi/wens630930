import os
import sys
import logging
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import timedelta, datetime
from tqdm import tqdm

# 移除当前目录
if 'd:\\data\\VSCode\\wens630930\\src\\dataset' in sys.path:
    sys.path.remove('d:\\data\\VSCode\\wens630930\\src\\dataset')
    # 获取项目根目录
    project_root = Path(__file__).resolve().parent.parent
    sys.path.append(str(project_root))

from dataset.base_dataset import BaseDataSet
from dataset.base_feature_dataset import BaseFeatureDataSet
import configs.base_config as config
from transform.features import FeatureType, FeatureDtype, Feature
from configs.base_config import RawData


# 显示所有列
pd.set_option('display.max_columns', None)
# 显示所有行
pd.set_option('display.max_rows', None)
# 设置value的显示长度为100，默认为50
pd.set_option('max_colwidth', 100)

base_dir = os.path.split(os.path.realpath(__file__))[0]
program = os.path.basename(sys.argv[0])
logger = logging.getLogger(program)
logging.basicConfig(format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s: %(message)s',
                    datefmt='%a, %d %b %Y %H:%M:%S')
logging.root.setLevel(level=logging.INFO)


class ImmuneFeature(BaseFeatureDataSet):


    def __init__(self, running_dt_end: str, train_interval: int, file_type: str, **param):
        super().__init__(param)
        logger.info('-----Loading data-----')
        # 加载生产数据
        self.production_data = pd.read_csv(RawData.ADS_PIG_ORG_TOTAL_TO_ML_TRAINING_DAY.value, encoding='utf-8-sig')
        # 加载免疫数据
        self.pigfarm_immune = pd.read_csv(RawData.TMP_CQ_IMMUPLANDT.value, encoding='utf-8-sig', low_memory=False)
        self.fuwubu_immune = pd.read_csv(RawData.ADS_CQ_FWB_IMMU_SMART_MANA_MDL.value, encoding='utf-8-sig')

        self.running_dt_end = running_dt_end
        self.train_interval = train_interval
        self.file_type = file_type

        self.end_date = (datetime.strptime(self.running_dt_end, "%Y-%m-%d") - timedelta(days=1)).strftime("%Y-%m-%d")
        self.start_date = (datetime.strptime(self.end_date, "%Y-%m-%d") - timedelta(days=self.train_interval) - timedelta(days=35)).strftime("%Y-%m-%d")
        logger.info('-----start_date: {}'.format(self.start_date))
        logger.info('-----end_date: {}'.format(self.end_date))

        self.data = pd.DataFrame()
        self.file_name = None

        self._init_entity_and_features()

    def _preprocessing_data(self):
        # 猪场免疫数据预处理
        pigfarm_immune = self.pigfarm_immune.copy()
        # 筛选掉exec_stat为'正常未免'的行
        pigfarm_immune = pigfarm_immune[pigfarm_immune['exec_status'] != '正常未免']
        # 筛选蓝耳病疫苗
        pigfarm_immune = pigfarm_immune[(pigfarm_immune['vaccine_type_nm'] == '蓝耳疫苗')]
        # 筛选group_type_nm
        sow_group = ['后备种猪', '待配后备猪', '隔离舍≥210天龄后备猪', '回场猪', '≥150且≤250天龄回场猪', '≥190天龄回场猪', '后备猪', '≥150天龄回场猪', '≥220天龄后备猪', '≥200天龄后备猪', '养户回场猪', '生产线后备母猪', '怀孕母猪（怀孕91天）', '断奶母猪', '怀孕≤85天龄母猪', '生产母猪', '哺乳', '怀孕≤70天龄母猪', '怀孕1胎母猪（怀孕77天）', '分娩', '待配', '生产线普免', '怀孕母猪(怀孕105天）', '配种', '断奶', '基础母猪', '妊娠', '怀孕母猪', '全场母猪公猪', '生产种猪']
        pigfarm_immune = pigfarm_immune[pigfarm_immune['group_type_nm'].isin(sow_group)]
        # 先将字符串'NULL'替换为np.nan
        pigfarm_immune['fac_end_dt'] = pigfarm_immune['fac_end_dt'].replace('NULL', np.nan)
        # 确保计划日期格式正确
        pigfarm_immune['plan_end_dt'] = pd.to_datetime(pigfarm_immune['plan_end_dt'], format='mixed')
        pigfarm_immune['fac_end_dt'] = pd.to_datetime(pigfarm_immune['fac_end_dt'], format='mixed', errors='coerce')

        # 服务部免疫数据预处理
        fuwubu_immune = self.fuwubu_immune.copy()
        # 筛选掉exec_stat为'正常未免'的行
        fuwubu_immune = fuwubu_immune[fuwubu_immune['exec_status'] != '正常未免']
        # 筛选饲养品种名称
        fuwubu_immune = fuwubu_immune[(fuwubu_immune['feed_breeds_nm'] == '种猪')]
        # 确保日期格式正确
        fuwubu_immune.rename(columns={'fplan_end_dt': 'plan_end_dt'}, inplace=True)
        fuwubu_immune['plan_end_dt'] = pd.to_datetime(fuwubu_immune['plan_end_dt'], format='mixed')
        fuwubu_immune['fac_end_dt'] = pd.to_datetime(fuwubu_immune['fac_end_dt'], format='mixed', errors='coerce')

        # 生产数据处理
        index_data = self.production_data.copy()
        index_data['stats_dt'] = pd.to_datetime(index_data['stats_dt'])
        index_data = index_data[(index_data['stats_dt'] >= self.start_date) & (index_data['stats_dt'] <= self.end_date)]

        # 更新
        self.immune_data = pd.concat([pigfarm_immune, fuwubu_immune], ignore_index=True)
        self.immune_data.sort_values(by=['org_inv_dk', 'plan_end_dt'], inplace=True)
        self.immune_data = self.immune_data[['org_inv_dk', 'plan_end_dt', 'fac_end_dt']].copy()
        self.index_data = index_data[['pigfarm_dk', 'stats_dt']].copy()
        self.index_data.sort_values(by=['pigfarm_dk', 'stats_dt'], inplace=True)

        self.data = self.index_data[['pigfarm_dk', 'stats_dt']].copy()

    def _init_entity_and_features(self):
        """初始化实体和特征定义"""
        # 设置实体列表
        self.entity = ['plan_end_dt', 'org_inv_dk']
        # 定义特征
        features_config = [
            # 免疫特征
            ('is_not_immu_21d', FeatureType.Categorical, FeatureDtype.Int32, 'is_not_immu_21d'),
            ('immu_count_30d', FeatureType.Continuous, FeatureDtype.Int32, 'immu_count_30d'),
            ('immu_avg_delay_days_30d', FeatureType.Continuous, FeatureDtype.Float32, 'immu_avg_delay_days_30d'),
            ('immu_delay_count_30d', FeatureType.Continuous, FeatureDtype.Int32, 'immu_delay_count_30d'),
            ('immu_delay_ratio_30d', FeatureType.Continuous, FeatureDtype.Float32, 'immu_delay_ratio_30d'),
        ]

        # 创建并添加特征到 features 对象
        for name, feature_type, dtype, domain in features_config:
            feature = Feature(
                name=name,
                domain=domain,
                feature_type=feature_type,
                dtype=dtype
            )
            self.features.add(feature)

        logger.info(f"初始化完成 - 实体数量: {len(self.entity)}, 特征数量: {len(self.features)}")


    def _get_is_not_immu_21d_feature(self):
        # 获取数据
        index_data = self.index_data.copy()
        # 获取fac_end_dt为空的数据，即未免疫的数据
        immune_data = self.immune_data[self.immune_data['fac_end_dt'].isna()].copy()

        # 初始化特征列
        index_data['is_not_immu_21d'] = 0

        # 对每个猪场分别处理
        for farm_dk in tqdm(index_data['pigfarm_dk'].unique(), desc="计算is_not_immu_21d特征"):
            # 获取该猪场的生产数据
            farm_date = index_data[index_data['pigfarm_dk'] == farm_dk]

            # 获取该猪场的免疫数据
            farm_immune = immune_data[immune_data['org_inv_dk'] == farm_dk]

            # 如果该猪场没有免疫记录则跳过
            if farm_immune.empty:
                continue

            # 对每条生产记录检查21天内是否有免疫
            for idx, prod_row in farm_date.iterrows():
                start_dt = prod_row['stats_dt']
                start_date = start_dt - pd.Timedelta(days=20)

                # 检查是否有免疫记录在21天窗口内
                immune_in_window = farm_immune[
                    (farm_immune['plan_end_dt'] <= start_dt) &
                    (farm_immune['plan_end_dt'] >= start_date)
                ]

                if not immune_in_window.empty:
                    index_data.loc[idx, 'is_not_immu_21d'] = 1

        self.data = self.data.merge(index_data[['stats_dt', 'pigfarm_dk', 'is_not_immu_21d']], on=['stats_dt', 'pigfarm_dk'], how='outer')

    def _get_immu_count_30d_feature(self):
        # 获取数据
        index_data = self.index_data.copy()
        immune_data = self.immune_data.copy()

        # 初始化特征列
        index_data['immu_count_30d'] = np.nan

        # 对每个猪场分别处理
        for farm_dk in tqdm(index_data['pigfarm_dk'].unique(), desc="计算immu_count_30d特征"):
            # 获取该猪场的生产数据
            farm_date = index_data[index_data['pigfarm_dk'] == farm_dk]

            # 获取该猪场的免疫数据
            farm_immune = immune_data[immune_data['org_inv_dk'] == farm_dk]

            # 如果该猪场没有免疫记录则跳过
            if farm_immune.empty:
                continue

            # 对每条生产记录计算30天内免疫数量
            for idx, prod_row in farm_date.iterrows():
                stats_dt = prod_row['stats_dt']
                start_date = stats_dt - pd.Timedelta(days=29)  # stats_dt-29到stats_dt，共30天

                # 筛选30天窗口内的免疫记录
                immune_in_window = farm_immune[
                    (farm_immune['plan_end_dt'] >= start_date) &
                    (farm_immune['plan_end_dt'] <= stats_dt)
                ]

                # 计算免疫记录数量
                index_data.loc[idx, 'immu_count_30d'] = len(immune_in_window)

        # 合并到主数据集
        self.data = self.data.merge(index_data[['stats_dt', 'pigfarm_dk', 'immu_count_30d']],
                                on=['stats_dt', 'pigfarm_dk'], how='left')

    def _get_immu_avg_delay_days_30d_feature(self):
        # 获取数据
        index_data = self.index_data.copy()
        immune_data = self.immune_data[self.immune_data['fac_end_dt'].notna()].copy()

        # 初始化特征列
        index_data['immu_avg_delay_days_30d'] = np.nan

        # 对每个猪场分别处理
        for farm_dk in tqdm(index_data['pigfarm_dk'].unique(), desc="计算immu_avg_delay_days_30d特征"):
            # 获取该猪场的生产数据
            farm_date = index_data[index_data['pigfarm_dk'] == farm_dk]

            # 获取该猪场的免疫数据（只保留fac_end_dt非空的记录）
            farm_immune = immune_data[immune_data['org_inv_dk'] == farm_dk].copy()

            # 如果该猪场没有有效的免疫记录则跳过
            if farm_immune.empty:
                continue

            # 计算延迟天数
            farm_immune['delay_days'] = (farm_immune['fac_end_dt'] - farm_immune['plan_end_dt']).dt.days

            # 对每条生产记录计算30天内的平均延迟天数
            for idx, prod_row in farm_date.iterrows():
                stats_dt = prod_row['stats_dt']
                start_date = stats_dt - pd.Timedelta(days=29)  # stats_dt-29到stats_dt，共30天

                # 筛选30天窗口内的免疫记录
                immune_in_window = farm_immune[
                    (farm_immune['plan_end_dt'] >= start_date) &
                    (farm_immune['plan_end_dt'] <= stats_dt)
                ]

                # 计算平均延迟天数
                if not immune_in_window.empty:
                    avg_delay = immune_in_window['delay_days'].mean()
                    index_data.loc[idx, 'immu_avg_delay_days_30d'] = avg_delay

        # 合并到主数据集
        self.data = self.data.merge(index_data[['stats_dt', 'pigfarm_dk', 'immu_avg_delay_days_30d']],
                                    on=['stats_dt', 'pigfarm_dk'], how='left')

    def _get_immu_delay_count_30d_feature(self):
        # 获取数据
        index_data = self.index_data.copy()
        immune_data = self.immune_data[self.immune_data['fac_end_dt'].notna()].copy()

        # 初始化特征列
        index_data['immu_delay_count_30d'] = np.nan

        # 对每个猪场分别处理
        for farm_dk in tqdm(index_data['pigfarm_dk'].unique(), desc="计算immu_delay_count_30d特征"):
            # 获取该猪场的生产数据
            farm_date = index_data[index_data['pigfarm_dk'] == farm_dk]

            # 获取该猪场的免疫数据（只保留fac_end_dt非空的记录）
            farm_immune = immune_data[immune_data['org_inv_dk'] == farm_dk].copy()

            # 如果该猪场没有有效的免疫记录则跳过
            if farm_immune.empty:
                continue

            # 对每条生产记录计算30天内的延迟免疫数量
            for idx, prod_row in farm_date.iterrows():
                stats_dt = prod_row['stats_dt']
                start_date = stats_dt - pd.Timedelta(days=29)  # stats_dt-29到stats_dt，共30天

                # 筛选30天窗口内的免疫记录
                immune_in_window = farm_immune[
                    (farm_immune['plan_end_dt'] >= start_date) &
                    (farm_immune['plan_end_dt'] <= stats_dt)
                ]

                # 计算延迟免疫的数量（fac_end_dt > plan_end_dt）
                if not immune_in_window.empty:
                    delay_count = len(immune_in_window[
                        immune_in_window['fac_end_dt'] > immune_in_window['plan_end_dt']
                    ])
                    index_data.loc[idx, 'immu_delay_count_30d'] = delay_count

        # 合并到主数据集
        self.data = self.data.merge(index_data[['stats_dt', 'pigfarm_dk', 'immu_delay_count_30d']],
                                    on=['stats_dt', 'pigfarm_dk'], how='left')

    def _get_immu_delay_ratio_30d_feature(self):
        # 计算延迟免疫比例：延迟免疫次数 / 总免疫次数
        # 当总免疫次数为0时，比例设为NaN
        self.data['immu_delay_ratio_30d'] = np.where(
            self.data['immu_count_30d'] > 0,
            self.data['immu_delay_count_30d'] / self.data['immu_count_30d'],
            np.nan
        )

    def _post_processing_data(self):
        if self.data.isnull().any().any():
            logger.info("Warning: Null in death_confirm_feature_data.csv")
        self.file_name = "death_confirm_feature_data." + self.file_type

        data = self.data.copy()

        index_cols = ['stats_dt', 'pigfarm_dk']
        immu_feature_list = [
            'is_not_immu_21d',
            'immu_count_30d',
            'immu_avg_delay_days_30d',
            'immu_delay_count_30d',
            'immu_delay_ratio_30d'
        ]
        data = data[index_cols + immu_feature_list]
        data['stats_dt'] = data['stats_dt'] + pd.DateOffset(days=1)  # 确保日期是正确的

        self.data = data.copy()

    def build_dataset_all(self):
        logger.info("-----Preprocessing data----- ")
        self._preprocessing_data()
        logger.info("Calculating interval from last purchase...")
        self._get_is_not_immu_21d_feature()
        self._get_immu_count_30d_feature()
        self._get_immu_avg_delay_days_30d_feature()
        self._get_immu_delay_count_30d_feature()
        self._get_immu_delay_ratio_30d_feature()
        logger.info("-----Postprocessing data----- ")
        self._post_processing_data()
        # logger.info("-----Save as : {}".format("/".join([config.FEATURE_STORE_ROOT, self.file_name])))
        logger.info("-----Save as : {}".format(config.FeatureData.IMMUNE_FEATURE_DATA.value))
        # self.dump_dataset("/".join([config.FEATURE_STORE_ROOT, self.file_name]))
        self.dump_dataset(config.FeatureData.IMMUNE_FEATURE_DATA.value)
        logger.info("-----Dataset saved successfully-----")

if __name__ == "__main__":
    running_dt_end = "2024-12-31"
    train_interval = 30
    file_type = "csv"
    immune_feature = ImmuneFeature(running_dt_end, train_interval, file_type)
    immune_feature.build_dataset_all()



