

def create_sequences(data, target_column='has_risk_label', seq_length=7, feature_columns=None):
    """
    构建LSTM模型的序列输入数据
    
    Args:
        data (DataFrame): 已转换和处理过的特征数据，包含掩码列
        target_column (str): 目标标签列名，默认为'has_risk_label'
        seq_length (int): 序列长度，默认为30天
        feature_columns (list): 要包含的特征列，如果为None则自动筛选
        
    Returns:
        tuple: (X, y) 特征序列和对应的标签
    """
    # 确保日期列是datetime类型
    data['stats_dt'] = pd.to_datetime(data['stats_dt'])
    
    # 如果未指定特征列，则排除非特征列
    if feature_columns is None:
        exclude_cols = ['stats_dt', target_column]
        feature_columns = [col for col in data.columns if col not in exclude_cols]
    
    logger.info(f"开始构建序列数据，序列长度: {seq_length}，目标列: {target_column}")
    logger.info(f"使用特征列数量: {len(feature_columns)}")
    
    X = []  # 特征序列
    y = []  # 标签

    feature_mask = []
    for feature in feature_columns:
        feature_mask.append(feature)
        feature_mask.append(feature + '_mask')
    
    # 按猪场ID分组处理
    farm_groups = data.groupby('pigfarm_dk')
    
    for farm_id, farm_data in tqdm(farm_groups, desc="构建序列"):
        # 确保数据按日期排序
        farm_data = farm_data.sort_values('stats_dt').reset_index(drop=True)
        
        # 从第seq_length条记录开始处理
        for i in range(seq_length, len(farm_data)):
            current_date = farm_data.iloc[i]['stats_dt']
            current_label = farm_data.iloc[i][target_column]
            
            # 获取当前日期前seq_length天的数据
            # 不包含当前日期的数据
            history = farm_data.iloc[i-seq_length:i]
            
            # 提取特征数据
            seq_features = history[feature_mask].values
            
            # 如果序列长度正确，添加到结果中
            if len(seq_features) == seq_length:
                X.append(seq_features)
                y.append(current_label)
    
    # 转换为numpy数组
    X = np.array(X)
    y = np.array(y)
    
    logger.info(f"序列构建完成，共生成 {len(X)} 个序列")
    logger.info(f"特征形状: {X.shape}, 标签形状: {y.shape}")
    
    return X, y