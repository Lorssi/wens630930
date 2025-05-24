from dataclasses import dataclass, field
from typing import List

from dataclasses_json import dataclass_json
import pandas as pd

from .features import FeatureDtype, Features, Feature, FeatureType
from .features import StringLookup, Normalization
from .base_transformer import BaseTransform

from configs.feature_config import ColumnsConfig

"""
鸡群均重预测模块输入特征：
- 品类l3_breeds_class_nm: Categorical类。
- 饲养品种feed_breeds_nm: Categorical类。
- 性别gender: Categorical类。
- 档次breeds_class_nm: Categorical类。
- 养户rearer_dk: Categorical类。
- 服务部org_inv_dk: Categorical类。
- 技术员tech_bk: Categorical类。
- 月份month: Categorical类。
- 天龄retrieve_days: Continuous类。

"""


@dataclass_json
@dataclass
class AbortionPredictionTransform(BaseTransform):
    
    features: Features = None  #必须有
    categorical_feature_names: List[str] = None
    continuous_feature_names: List[str] = None
    unchanged_feature_names: List[str] = None
    transform_feature_names: List[str] = None

    def __post_init__(self):
        # maybe do something
        self.categorical_feature_names = ColumnsConfig.DISCRETE_COLUMNS
        self.continuous_feature_names = ColumnsConfig.CONTINUOUS_COLUMNS
        self.unchanged_feature_names = ColumnsConfig.INVARIANT_COLUMNS

    def _fit_categorical_feature(self, feature_name: str, input_dataset: pd.DataFrame, column_name: str):
        feature = Feature(name=feature_name, feature_type=FeatureType.Categorical)
        # 注意编码要从1开始，0代表未见的id。获取长度的时候记得要len(feature.category_encode)+1
        feature.category_encode = StringLookup(name=feature_name, offset=1)
        feature.category_encode.fit(input_dataset[column_name].unique().tolist())
        self.features.add(feature)

    def _transform_categorical_feature(self, feature_name: str, input_dataset: pd.DataFrame, column_name: str, output_dataset: pd.DataFrame):
        feature = self.features[feature_name]
        # transform_series = input_dataset[column_name].map(feature.category_encode.code_book).rename(column_name)
        transform_series = feature.category_encode.transform_series(input_dataset[column_name], default=0).rename(column_name)
        new_df = pd.DataFrame(transform_series)
        return pd.concat([output_dataset.reset_index(drop=True), new_df.reset_index(drop=True)], axis=1)

    def _fit_transform_categorical_feature(self, feature_name: str, input_dataset: pd.DataFrame, column_name: str, output_dataset: pd.DataFrame):
        self._fit_categorical_feature(feature_name, input_dataset, column_name)
        return self._transform_categorical_feature(feature_name, input_dataset, column_name, output_dataset=output_dataset)

    def _fit_continuous_feature(self, feature_name: str, input_dataset: pd.DataFrame, column_name: str):
        feature = Feature(name=feature_name, feature_type=FeatureType.Continuous)
        feature.normalization = Normalization(name=feature_name)
        feature.normalization.fit(input_dataset[column_name])
        self.features.add(feature)

    def _transform_continuous_feature(self, feature_name: str, input_dataset: pd.DataFrame, column_name: str, output_dataset: pd.DataFrame):
        feature = self.features[feature_name]
        transform_series = feature.normalization.transform(input_dataset[column_name]).rename(column_name)
        new_df = pd.DataFrame(transform_series)
        return pd.concat([output_dataset.reset_index(drop=True), new_df.reset_index(drop=True)], axis=1)

    def _fit_transform_continuous_feature(self, feature_name: str, input_dataset: pd.DataFrame, column_name: str, output_dataset: pd.DataFrame):
        self._fit_continuous_feature(feature_name, input_dataset, column_name)
        return self._transform_continuous_feature(feature_name, input_dataset, column_name, output_dataset=output_dataset)

    def fit_transform(self, input_dataset: pd.DataFrame):
        # do something, 实现具体逻辑
        output_dataframe = pd.DataFrame()
        for name in self.categorical_feature_names:
            output_dataframe = self._fit_transform_categorical_feature(feature_name=name, input_dataset=input_dataset, column_name=name, output_dataset=output_dataframe)
        for name in self.continuous_feature_names:
            output_dataframe = self._fit_transform_continuous_feature(feature_name=name, input_dataset=input_dataset, column_name=name, output_dataset=output_dataframe)
        for name in self.unchanged_feature_names:
            output_dataframe[name] = input_dataset[name]

        output_dataframe = output_dataframe[self.transform_feature_names]

        return output_dataframe

    def transform(self, input_dataset: pd.DataFrame):
        # do something, 实现具体逻辑
        output_dataframe = pd.DataFrame()
        for name in self.categorical_feature_names:
            output_dataframe = self._transform_categorical_feature(feature_name=name, input_dataset=input_dataset, column_name=name, output_dataset=output_dataframe)
        for name in self.continuous_feature_names:
            output_dataframe = self._transform_continuous_feature(feature_name=name, input_dataset=input_dataset, column_name=name, output_dataset=output_dataframe)
        for name in self.unchanged_feature_names:
            output_dataframe[name] = input_dataset[name]

        output_dataframe = output_dataframe[self.transform_feature_names]

        return output_dataframe

@dataclass_json
@dataclass
class AbortionPredictionTransformPipeline(BaseTransform):
    features: Features = None
    trans: AbortionPredictionTransform = None
    transform_feature_names: List[str] = None

    def __post_init__(self):
        if self.trans is None:
            self.trans = AbortionPredictionTransform(features=Features(), transform_feature_names=self.transform_feature_names)

    def fit_transform(self, input_dataset: pd.DataFrame):
        X1:pd.DataFrame = self.trans.fit_transform(input_dataset)
        self.features = self.trans.features # 获得transform
        return X1

    def transform(self, input_dataset: pd.DataFrame):
        X1:pd.DataFrame = self.trans.transform(input_dataset)
        return X1
    