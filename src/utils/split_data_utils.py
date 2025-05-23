






# def split_data(data_transformed_masked_null, y):
#     """
#     按日期百分比划分训练集和验证集
    
#     Args:
#         data (DataFrame): 包含日期列的数据
#         test_split_ratio (float): 验证集占比，默认0.2
        
#     Returns:
#         tuple: (训练集数据, 验证集数据)
#     """
#     # 确保日期列是datetime类型
#     data_transformed_masked_null = data_transformed_masked_null.copy()
#     y = y.copy()

#     data_transformed_masked_null['stats_dt'] = pd.to_datetime(data_transformed_masked_null['stats_dt'])
#     y['stats_dt'] = pd.to_datetime(y['stats_dt'])
    
#     # 获取唯一的日期列表并排序
#     unique_dates = sorted(data_transformed_masked_null['stats_dt'].unique())
    
#     # 计算分割点
#     split_idx = int(len(unique_dates) * (1 - config.TEST_SPLIT_RATIO))
#     split_date = unique_dates[split_idx]
    
#     logger.info(f"按日期百分比划分数据: 分割日期 = {split_date}")
    
#     # 划分数据
#     train_data_X = data_transformed_masked_null[data_transformed_masked_null['stats_dt'] < split_date]
#     train_data_y = y[y['stats_dt'] < split_date]
#     val_data_X = data_transformed_masked_null[data_transformed_masked_null['stats_dt'] >= split_date]
#     val_data_y = y[y['stats_dt'] >= split_date]
    
#     # 记录划分结果
#     logger.info(f"训练集: {len(train_data_X)}条记录, 日期范围: {train_data_X['stats_dt'].min()} - {train_data_X['stats_dt'].max()}")
#     logger.info(f"验证集: {len(val_data_X)}条记录, 日期范围: {val_data_X['stats_dt'].min()} - {val_data_X['stats_dt'].max()}")

#     train_data_X = train_data_X[ColumnsConfig.feature_columns]
#     val_data_X = val_data_X[ColumnsConfig.feature_columns]
#     train_data_y = train_data_y[ColumnsConfig.HAS_RISK_LABEL]
#     val_data_y = val_data_y[ColumnsConfig.HAS_RISK_LABEL]
    
#     return train_data_X, val_data_X, train_data_y, val_data_y