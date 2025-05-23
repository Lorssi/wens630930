# utils/logger.py
import logging
import os

def setup_logger(log_file_path, logger_name='MyLogger', level=logging.INFO):
    """设置一个简单的日志记录器"""
    # 确保日志目录存在
    log_dir = os.path.dirname(log_file_path)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir)

    logger = logging.getLogger(logger_name)
    logger.setLevel(level)

    # 防止重复添加handler
    if not logger.handlers:
        # 文件处理器
        fh = logging.FileHandler(log_file_path)
        fh.setLevel(level)
        # 控制台处理器
        ch = logging.StreamHandler()
        ch.setLevel(level)

        # 创建格式器并将其添加到处理器
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)

        # 将处理器添加到记录器
        logger.addHandler(fh)
        logger.addHandler(ch)
    return logger