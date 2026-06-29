"""
统一日志配置

使用场景:
- 开发: DEBUG 级别, 彩色输出到控制台
- 生产: INFO 级别, 输出到文件和 journald

Logger 层级:
  所有模块使用 get_logger(__name__), 返回 "smart_speaker.xxx" 格式的 logger,
  统一继承自 "smart_speaker" 根 logger。调用 setup_logger 后, 所有子 logger
  自动继承根 logger 的级别和 handlers。
"""

import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Optional

ROOT_LOGGER_NAME = "smart_speaker"


def setup_logger(
    name: str = ROOT_LOGGER_NAME,
    level: str = "INFO",
    log_dir: Optional[str] = None,
    to_file: bool = True,
    to_console: bool = True,
) -> logging.Logger:
    """
    配置根 logger (smart_speaker) 的 handlers 和级别。
    只应调用一次。后续调用会更新级别但不重复添加 handlers。

    Args:
        name: logger 名称 (默认 smart_speaker)
        level: 日志级别 (DEBUG|INFO|WARNING|ERROR)
        log_dir: 日志文件目录 (默认 ./data/logs)
        to_file: 是否输出到文件
        to_console: 是否输出到控制台
    """
    logger = logging.getLogger(name)

    # 已有 handler 时: 不覆盖级别 (保留已设置的, 如命令行 --log-level),
    # 但重置子 logger 让它们继承根 logger
    if logger.handlers:
        _reset_child_levels(name)
        return logger

    logger.setLevel(getattr(logging, level.upper(), logging.INFO))

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

    # 抑制第三方库的 DEBUG 日志
    for lib in ("urllib3", "requests", "httpx", "httpcore", "openwakeword",
                "aiohttp", "asyncio", "edge_tts"):
        logging.getLogger(lib).setLevel(logging.WARNING)

    # 抑制 C 库直接输出的噪音（ALSA Unknown PCM、Vosk Kaldi LOG）
    suppress_native_logs()

    return logger


def get_logger(name: str = ROOT_LOGGER_NAME) -> logging.Logger:
    """
    获取子 logger。所有模块应使用 get_logger(__name__),
    返回的 logger 会继承 smart_speaker 根 logger 的配置。

    例如: get_logger("src.wake_word.detector") → logger "smart_speaker.src.wake_word.detector"
    """
    if name == ROOT_LOGGER_NAME or name.startswith(ROOT_LOGGER_NAME + "."):
        return logging.getLogger(name)
    return logging.getLogger(f"{ROOT_LOGGER_NAME}.{name}")


def suppress_native_logs() -> None:
    """抑制 C 库（Vosk/Kaldi）直接输出到 stderr 的噪音日志"""
    # Vosk/Kaldi: 设置最低日志级别消除 "LOG (VoskAPI:...)" 输出
    try:
        import vosk
        vosk.SetLogLevel(-1)
    except Exception:
        pass


def suppress_alsa_noise() -> None:
    """初始化音频模块时临时静默 ALSA stderr 噪音 (Unknown PCM 等无害警告)"""
    import os
    devnull = os.open(os.devnull, os.O_WRONLY)
    old_stderr = os.dup(2)
    os.dup2(devnull, 2)
    os.close(devnull)
    return old_stderr


def restore_alsa_noise(old_stderr: int) -> None:
    """恢复 stderr"""
    import os
    os.dup2(old_stderr, 2)
    os.close(old_stderr)


def _reset_child_levels(root_name: str) -> None:
    """重置所有子 logger 的级别为 NOTSET, 让它们继承根 logger"""
    prefix = root_name + "."
    for name in logging.root.manager.loggerDict:
        if name.startswith(prefix):
            child = logging.getLogger(name)
            if child.level != logging.NOTSET:
                child.setLevel(logging.NOTSET)
