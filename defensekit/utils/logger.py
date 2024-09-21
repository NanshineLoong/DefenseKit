import logging
import os
import time

def setup_logger(logging_path: str) -> logging.Logger:
    """
    设置一个日志记录器，包括文件和控制台输出。

    Args:
        logging_path (str): 日志文件的存放路径。

    Returns:
        logging.Logger: 配置好的日志记录器对象。
    """
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    # 创建文件处理器
    time_str = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    log_file = os.path.join(logging_path, f"{time_str}.log")
    if not os.path.exists(logging_path):
        os.makedirs(logging_path)
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # 创建控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

