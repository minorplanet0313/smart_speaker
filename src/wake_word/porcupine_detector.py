"""
Porcupine Wake Word Detector

基于 Picovoice Porcupine v4，支持:
- 内置英文唤醒词: \"porcupine\", \"computer\", \"jarvis\", \"alexa\" 等
- 自定义中文唤醒词: 通过 Picovoice Console 生成的 .ppn 文件

准确率 ~97%，延迟 < 150ms，资源占用极低。
Picovoice Console: https://console.picovoice.ai/ (免费注册，获取 access_key)

对比 openWakeWord:
    - 更高准确率 (97% vs ~85%)
    - 更低资源占用 (<500KB RAM)
    - 原生中文支持
"""

import os
import threading
import time
from typing import Callable, Optional

import numpy as np

from src.utils.logger import get_logger

logger = get_logger(__name__)


class PorcupineDetector:
    """Porcupine 唤醒词检测器"""

    def __init__(
        self,
        access_key: str = "",
        keyword: str = "porcupine",
        keyword_path: str = "",
        sensitivity: float = 0.7,
        model_path: str = "",
        sample_rate: int = 16000,
    ):
        """
        Args:
            access_key: Picovoice Access Key (免费注册: https://console.picovoice.ai/)
            keyword: 内置唤醒词 (如 "porcupine", "computer", "jarvis")
            keyword_path: 自定义 .ppn 模型文件路径 (如 "xiaozhi_zh.ppn")
            sensitivity: 灵敏度 [0, 1], 越高越灵敏但也可能误触发
            model_path: 自定义模型参数文件 (.pv)
            sample_rate: 采样率 (Porcupine 仅支持 16000)
        """
        self.access_key = access_key
        self.keyword = keyword
        self.keyword_path = keyword_path
        self.sensitivity = sensitivity
        self.model_path = model_path
        self.sample_rate = sample_rate

        self._porcupine = None
        self._init_attempted = False  # 防止每次 detect() 都重试初始化
        self._callbacks: list[Callable[[float], None]] = []
        self._frame_length: int = 0

    def _lazy_init(self) -> None:
        """延迟初始化 Porcupine (仅尝试一次，失败后不再重试)"""
        if self._porcupine is not None or self._init_attempted:
            return
        self._init_attempted = True
        try:
            import pvporcupine

            kwargs = {"sensitivities": [self.sensitivity]}

            if self.keyword_path:
                # 使用自定义 .ppn 模型文件
                if not os.path.exists(self.keyword_path):
                    logger.warning(f"Porcupine 自定义模型不存在: {self.keyword_path}")
                    return
                kwargs["keyword_paths"] = [self.keyword_path]
                logger.info(f"Porcupine 加载自定义模型: {self.keyword_path}")
            elif self.keyword:
                # 使用内置唤醒词
                kwargs["keywords"] = [self.keyword]
                logger.info(f"Porcupine 使用内置唤醒词: {self.keyword}")

            if self.access_key:
                kwargs["access_key"] = self.access_key

            if self.model_path:
                kwargs["model_path"] = self.model_path

            self._porcupine = pvporcupine.create(**kwargs)
            self._frame_length = self._porcupine.frame_length
            logger.info(
                f"Porcupine 初始化完成: sensitivity={self.sensitivity}, "
                f"frame_length={self._frame_length}"
            )

        except ImportError:
            logger.warning("pvporcupine 未安装: pip install pvporcupine")
        except Exception as e:
            error_msg = str(e)
            if "access_key" in error_msg.lower() or "activation" in error_msg.lower():
                logger.warning(
                    f"Porcupine Access Key 无效或缺失。\n"
                    f"  免费获取: https://console.picovoice.ai/\n"
                    f"  错误: {e}"
                )
            else:
                logger.error(f"Porcupine 初始化失败: {e}")

    def detect(self, audio_frame: np.ndarray) -> float:
        """
        处理音频帧，检测唤醒词

        Args:
            audio_frame: float32 数组, 16kHz 单声道, range=[-1, 1]

        Returns:
            置信度分数 [0, 1], 0 表示未检测到
        """
        self._lazy_init()
        if self._porcupine is None:
            return 0.0

        # 转换为 int16 (Porcupine 要求)
        int16_frame = (np.clip(audio_frame, -1.0, 1.0) * 32767).astype(np.int16)

        # 确保帧长度匹配 (Porcupine frame_length 通常是 512)
        if len(int16_frame) < self._frame_length:
            # 补零
            padded = np.zeros(self._frame_length, dtype=np.int16)
            padded[:len(int16_frame)] = int16_frame
            int16_frame = padded
        elif len(int16_frame) > self._frame_length:
            int16_frame = int16_frame[:self._frame_length]

        result = self._porcupine.process(int16_frame)
        if result >= 0:
            confidence = 1.0  # Porcupine 返回 -1 或 keyword_index
            self._on_trigger(confidence)
            return confidence

        return 0.0

    def _on_trigger(self, confidence: float) -> None:
        """唤醒词触发"""
        logger.info(f"🔊 Porcupine 唤醒! keyword={self.keyword}")
        for callback in self._callbacks:
            try:
                callback(confidence)
            except Exception as e:
                logger.error(f"唤醒回调异常: {e}")

    def on_detected(self, callback: Callable[[float], None]) -> None:
        """注册唤醒回调"""
        self._callbacks.append(callback)

    def set_sensitivity(self, sensitivity: float) -> None:
        """动态调整灵敏度"""
        self.sensitivity = max(0.0, min(1.0, sensitivity))
        logger.info(f"Porcupine 灵敏度调整为: {self.sensitivity}")

    @property
    def is_available(self) -> bool:
        """Porcupine 是否可用"""
        self._lazy_init()
        return self._porcupine is not None

    def release(self) -> None:
        """释放资源"""
        if self._porcupine:
            try:
                self._porcupine.delete()
            except Exception:
                pass
            self._porcupine = None
