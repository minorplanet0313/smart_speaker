"""
统一日志配置

使用场景:
- 开发: DEBUG 级别, 彩色输出到控制台
- 生产: INFO 级别, 输出到文件和 journald
"""

import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Optional


def setup_logger(
    name: str = "smart_speaker",
    level: str = "INFO",
    log_dir: Optional[str] = None,
    to_file: bool = True,
    to_console: bool = True,
) -> logging.Logger:
    """
    配置并返回 logger

    Args:
        name: logger 名称
        level: 日志级别 (DEBUG|INFO|WARNING|ERROR)
        log_dir: 日志文件目录 (默认 ./data/logs)
        to_file: 是否输出到文件
        to_console: 是否输出到控制台
    """
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    # 避免重复添加 handler
    if logger.handlers:
        return logger

    # 日志格式
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # 控制台输出
    if to_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    # 文件输出 (轮转)
    if to_file:
        if log_dir is None:
            log_dir = "data/logs"
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        file_handler = RotatingFileHandler(
            filename=f"{log_dir}/smart_speaker.log",
            maxBytes=10 * 1024 * 1024,  # 10MB
            backupCount=5,
            encoding="utf-8",
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def get_logger(name: str = "smart_speaker") -> logging.Logger:
    """获取 logger (确保已初始化)"""
    logger = logging.getLogger(name)
    if not logger.handlers:
        return setup_logger(name)
    return logger
