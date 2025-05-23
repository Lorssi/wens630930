from pathlib import Path

class logger_config:
    """
    Logger configuration class.
    """
    ROOT_DIR = Path(__file__).parent.parent.parent
    LOG_DIR = ROOT_DIR / "logs"

    TRAIN_LOG_FILE_PATH = LOG_DIR / "training.log" # 日志文件路径 (如果使用logger)
    PREDICT_LOG_FILE_PATH = LOG_DIR / "predict.log" # 预测日志文件路径