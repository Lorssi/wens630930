import os
import pandas as pd
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

def save_to_csv(df, base_dir=None, filename=None, filepath=None, index=False, encoding='utf-8-sig'):
    """
    跨平台保存 DataFrame 到 CSV 文件
    
    Args:
        df (pandas.DataFrame): 要保存的数据帧
        base_dir (str, optional): 基础目录路径
        filename (str, optional): 文件名（包含或不包含.csv后缀）
        filepath (str, optional): 完整文件路径，如果提供此参数，将忽略 base_dir 和 filename
        index (bool): 是否保存索引，默认为False
        encoding (str): 文件编码，默认为'utf-8-sig'以支持中文
        
    Returns:
        str: 保存的文件完整路径
    """
    try:
        # 处理完整路径情况
        if filepath:
            # 支持 Path 对象或字符串
            path_obj = Path(filepath)
            base_dir = str(path_obj.parent)
            filename = path_obj.name
        elif not (base_dir and filename):
            raise ValueError("必须提供 filepath 或同时提供 base_dir 和 filename")
            
        # 确保文件名有.csv后缀
        if not filename.endswith('.csv'):
            filename = f"{filename}.csv"
        
        # 使用 Path 确保路径分隔符正确
        save_path = Path(base_dir) / filename
        
        # 创建目录（如果不存在）
        os.makedirs(Path(base_dir), exist_ok=True)
        
        # 保存CSV文件
        df.to_csv(save_path, index=index, encoding=encoding)
        
        logger.info(f"文件已保存到: {save_path}")
        return str(save_path)
    
    except Exception as e:
        logger.error(f"保存CSV文件时出错: {str(e)}")
        raise
        
def append_to_csv(df, filepath, index=False, encoding='utf-8-sig'):
    """
    向现有CSV文件追加数据
    
    Args:
        df (pandas.DataFrame): 要追加的数据帧
        filepath (str): 完整的文件路径
        index (bool): 是否保存索引，默认为False
        encoding (str): 文件编码，默认为'utf-8-sig'
        
    Returns:
        bool: 操作是否成功
    """
    try:
        # 将路径转换为平台特定格式
        filepath = Path(filepath)
        
        # 如果文件存在，追加模式
        if filepath.exists():
            df.to_csv(filepath, mode='a', header=False, index=index, encoding=encoding)
            logger.info(f"数据已追加到: {filepath}")
        else:
            # 如果文件不存在，创建新文件
            # 确保目录存在
            os.makedirs(filepath.parent, exist_ok=True)
            df.to_csv(filepath, index=index, encoding=encoding)
            logger.info(f"新文件已创建: {filepath}")
        
        return True
        
    except Exception as e:
        logger.error(f"追加到CSV文件时出错: {str(e)}")
        return False

def read_csv(filepath, encoding='utf-8-sig', **kwargs):
    """
    跨平台读取CSV文件
    
    Args:
        filepath (str): 文件路径
        encoding (str): 文件编码，默认为'utf-8-sig'
        **kwargs: 传递给pd.read_csv的其他参数
        
    Returns:
        pandas.DataFrame: 读取的数据帧
    """
    try:
        # 将路径转换为平台特定格式
        filepath = Path(filepath)
        
        if not filepath.exists():
            logger.error(f"文件不存在: {filepath}")
            return None
            
        return pd.read_csv(filepath, encoding=encoding, **kwargs)
        
    except Exception as e:
        logger.error(f"读取CSV文件时出错: {str(e)}")
        return None

