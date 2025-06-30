import pandas as pd
import os
import shutil
import glob


# 定义数据目录路径
data_dir = 'data/tmp'
version = 'v1.0.t'

# 从tmp目录中提取pigfarm_dk列表
def get_pigfarm_dks_from_files(data_dir):
    pigfarm_dks = []
    if os.path.exists(data_dir):
        for filename in os.listdir(data_dir):
            if filename.startswith('abort_abnormal_') and filename.endswith('.csv'):
                # 提取文件名中的pigfarm_dk部分
                pigfarm_dk = filename[len('abort_abnormal_'):-len('.csv')]
                pigfarm_dks.append(pigfarm_dk)
    return pigfarm_dks

pigfarm_dks = get_pigfarm_dks_from_files(data_dir)
pd.DataFrame(pigfarm_dks, columns=['pigfarm_dk']).to_csv('pigfarm_dks.csv', index=False)

pigfarm_dks = [dk.replace('/', '@') for dk in pigfarm_dks]

# 目标目录
predict_dir = 'data/predict/abort_abnormal'

# 获取 tmp 目录下的所有 CSV 文件
csv_files = glob.glob(os.path.join(data_dir, '*.csv'))

if csv_files:
    # 读取并合并所有 CSV 文件
    all_dataframes = []
    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file)
            all_dataframes.append(df)
            print(f'已读取文件: {csv_file}')
        except Exception as e:
            print(f'读取文件失败: {csv_file}, 错误: {e}')

    if all_dataframes:
        # 合并所有数据框
        combined_df = pd.concat(all_dataframes, ignore_index=True)

        # 创建目标目录结构：v1.0.t train/v1.0.t train 12/
        version_dir = f'{version} train'
        sub_dir = f'{version} train 12'
        full_target_dir = os.path.join(predict_dir, version_dir, sub_dir)

        # 创建目录
        os.makedirs(full_target_dir, exist_ok=True)

        # 保存合并后的文件
        target_file = os.path.join(full_target_dir, 'abort_abnormal.csv')
        combined_df.to_csv(target_file, index=False)

        print(f'已合并 {len(csv_files)} 个文件')
        print(f'合并后数据行数: {len(combined_df)}')
        print(f'保存至: {target_file}')
    else:
        print('没有成功读取任何 CSV 文件')
else:
    print(f'在 {data_dir} 目录下未找到任何 CSV 文件')

print('文件合并处理完成！')


# 处理每个 pigfarm_dk
for pigfarm_dk in pigfarm_dks:
    # 构建源文件路径
    source_file = os.path.join(data_dir, f'abort_abnormal_{pigfarm_dk}.csv')

    # 检查源文件是否存在
    if os.path.exists(source_file):
        # 创建版本目录结构：{version} {pigfarm_dk}/{version} {pigfarm_dk} 12/
        version_dir = f'{version} {pigfarm_dk}'
        sub_dir = f'{version} {pigfarm_dk} 12'
        full_target_dir = os.path.join(version_dir, sub_dir)

        # 创建目录
        os.makedirs(full_target_dir, exist_ok=True)

        # 移动并重命名文件
        target_file = os.path.join(full_target_dir, 'abort_abnormal.csv')
        shutil.move(source_file, target_file)
        print(f'已移动文件: {source_file} -> {target_file}')

        # 将版本目录移动到最终目标位置
        final_target = os.path.join(predict_dir, version_dir)
        if os.path.exists(final_target):
            shutil.rmtree(final_target)  # 如果目标目录已存在，先删除
        shutil.move(version_dir, final_target)
        print(f'已移动目录: {version_dir} -> {final_target}')
    else:
        print(f'警告: 源文件不存在 - {source_file}')

print('所有文件处理完成！')