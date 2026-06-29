"""
配置加载器

从 YAML 文件加载配置, 支持环境变量替换 ${VAR_NAME}
"""

import os
import re
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

try:
    from dotenv import load_dotenv
    _HAS_DOTENV = True
except ImportError:
    _HAS_DOTENV = False


class Config:
    """配置管理器, 支持点号路径访问和环境变量替换"""

    def __init__(self, config_path: Optional[str] = None):
        self._data: Dict[str, Any] = {}
        self._config_path = config_path

        # 加载 .env 文件 (可选依赖)
        if _HAS_DOTENV:
            env_file = Path(".env")
            if env_file.exists():
                load_dotenv(env_file, override=True)

        if config_path:
            self.load(config_path)

    def load(self, config_path: str) -> "Config":
        """从 YAML 文件加载配置"""
        with open(config_path, "r", encoding="utf-8") as f:
            raw = f.read()

        # 替换环境变量 ${VAR_NAME}
        raw = self._substitute_env(raw)

        self._data = yaml.safe_load(raw) or {}
        self._config_path = config_path
        return self

    def _substitute_env(self, text: str) -> str:
        """替换 ${VAR_NAME} 为环境变量值"""
        pattern = re.compile(r'\$\{(\w+)\}')

        def _replace(match):
            var_name = match.group(1)
            value = os.environ.get(var_name, "")
            if not value:
                # 只在 DEBUG 级别提示，避免每次启动都打印警告
                import logging
                logging.getLogger("smart_speaker").debug(
                    f"环境变量 {var_name} 未设置, 使用空字符串"
                )
            return value

        return pattern.sub(_replace, text)

    def get(self, key_path: str, default: Any = None) -> Any:
        """
        通过点号路径获取配置值
        例如: config.get("audio.sample_rate", 16000)
        """
        keys = key_path.split(".")
        value = self._data
        for key in keys:
            if isinstance(value, dict):
                value = value.get(key)
                if value is None:
                    return default
            else:
                return default
        return value

    def set(self, key_path: str, value: Any) -> None:
        """通过点号路径设置配置值"""
        keys = key_path.split(".")
        data = self._data
        for key in keys[:-1]:
            if key not in data:
                data[key] = {}
            data = data[key]
        data[keys[-1]] = value

    @property
    def data(self) -> Dict[str, Any]:
        return self._data

    def to_dict(self) -> Dict[str, Any]:
        return dict(self._data)

    def update_path(self, key_path: str, value: Any) -> None:
        """按 dot-path 更新单个配置值 (仅内存, 不写磁盘)"""
        self.set(key_path, value)

    def reload(self) -> None:
        """从磁盘重新加载配置，覆盖内存中的所有修改"""
        if self._config_path:
            self.load(self._config_path)

    def __repr__(self) -> str:
        return f"Config(path={self._config_path})"


# 全局单例
_config_instance: Optional[Config] = None


def get_config(config_path: Optional[str] = None) -> Config:
    """获取全局配置单例"""
    global _config_instance
    if _config_instance is None:
        if config_path is None:
            config_path = os.environ.get(
                "SMART_SPEAKER_CONFIG",
                "config/config.yaml"
            )
        _config_instance = Config(config_path)
    return _config_instance
