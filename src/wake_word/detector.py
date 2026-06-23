"""
唤醒词检测模块

基于 openWakeWord 框架
- Apache 2.0 开源协议
- 在 Pi 3 单核可运行 15-20 个模型
- 支持自定义唤醒词 (提供 Colab 训练脚本)

使用方式:
    detector = WakeWordDetector(model_path="xiao_zhi.onnx")
    detector.on_detected(lambda conf: print(f"Wake! {conf}"))

    # 每个音频帧
    score = detector.detect(audio_frame)
"""

import threading
import time
from typing import Callable, Optional

import numpy as np

from src.utils.logger import get_logger

logger = get_logger(__name__)


class WakeWordDetector:
    """
    唤醒词检测器

    内部维护一个滑动窗口, 将音频帧累积到足够长度后,
    送入 openWakeWord 模型进行推理。

    特性:
    - 支持 ONNX / TFLite 两种推理框架
    - 触发冷却 (防止重复触发)
    - 可动态调整阈值
    """

    def __init__(
        self,
        model_path: str,
        threshold: float = 0.5,
        inference_framework: str = "onnx",
        chunk_duration_ms: int = 80,
    ):
        """
        Args:
            model_path: 唤醒词模型文件路径 (.onnx 或 .tflite)
            threshold: 检测阈值 [0, 1], 越高误触发越少
            inference_framework: "onnx" 或 "tflite"
            chunk_duration_ms: 每个推理块的时长 (ms)
        """
        self.model_path = model_path
        self.threshold = threshold
        self.inference_framework = inference_framework
        self.chunk_duration_ms = chunk_duration_ms

        # 内部状态
        self._model = None
        self._callbacks: list[Callable[[float], None]] = []
        self._audio_buffer = np.array([], dtype=np.float32)
        self._samples_per_chunk = int(16000 * chunk_duration_ms / 1000)
        self._last_trigger_time = 0.0

        # 自检
        self._init_model()

    def _init_model(self) -> None:
        """初始化 openWakeWord 模型"""
        try:
            from openwakeword import Model

            if self.model_path and self.model_path != "":
                # 使用自定义模型
                self._model = Model(
                    wakeword_models=[self.model_path],
                    inference_framework=self.inference_framework,
                )
                logger.info(f"唤醒词模型加载完成: {self.model_path}")
            else:
                # 使用预置模型 (需要下载)
                logger.warning("未指定唤醒词模型路径, 将尝试使用内置模型")
                # openWakeWord 内置了一些预训练模型
                # 例如: "alexa", "hey_jarvis", "hey_mycroft" 等
                # 自定义模型需要使用 Colab 训练并导出
                try:
                    self._model = Model(
                        wakeword_models=["alexa"],  # 临时占位
                        inference_framework=self.inference_framework,
                    )
                    logger.info("使用内置唤醒词模型 (alexa 占位, 请替换)")
                except Exception:
                    self._model = None
                    logger.warning("唤醒词模型加载失败, 语音唤醒不可用")

        except ImportError:
            logger.warning("openwakeword 未安装, 语音唤醒不可用")
            logger.warning("安装: pip install openwakeword")
            self._model = None
        except Exception as e:
            logger.error(f"唤醒词模型初始化失败: {e}")
            self._model = None

    def detect(self, audio_frame: np.ndarray) -> float:
        """
        处理一个音频帧, 检测唤醒词

        Args:
            audio_frame: float32 数组, 16kHz 单声道, range=[-1, 1]

        Returns:
            置信度分数 [0, 1], 0 表示未检测到
        """
        if self._model is None:
            return 0.0

        # 累积音频缓冲
        self._audio_buffer = np.append(self._audio_buffer, audio_frame)

        # 等累积到足够的样本数再做推理
        if len(self._audio_buffer) >= self._samples_per_chunk:
            # 取出一个 chunk
            chunk = self._audio_buffer[:self._samples_per_chunk]
            self._audio_buffer = self._audio_buffer[self._samples_per_chunk:]

            try:
                # openWakeWord 推理
                # 输入要求: shape=(n_models, 1280) 类型的 float32
                # 对于 80ms chunk @ 16kHz: 1280 samples
                predictions = self._model.predict(chunk)

                # predictions 是 dict: {model_name: score}
                for model_name, score in predictions.items():
                    if score > self.threshold:
                        self._on_trigger(score)
                        return score
            except Exception as e:
                logger.error(f"唤醒词推理失败: {e}")

        return 0.0

    def _on_trigger(self, confidence: float) -> None:
        """唤醒词触发"""
        now = time.time()
        # 冷却检查 (至少间隔 2 秒)
        if now - self._last_trigger_time < 2.0:
            return

        self._last_trigger_time = now
        logger.info(f"唤醒词触发! confidence={confidence:.3f}")

        # 通知所有回调
        for callback in self._callbacks:
            try:
                callback(confidence)
            except Exception as e:
                logger.error(f"唤醒回调异常: {e}")

    def on_detected(self, callback: Callable[[float], None]) -> None:
        """
        注册唤醒回调

        Args:
            callback: 接收置信度分数的回调函数
        """
        self._callbacks.append(callback)

    def set_threshold(self, threshold: float) -> None:
        """动态调整检测阈值"""
        self.threshold = max(0.0, min(1.0, threshold))
        logger.info(f"唤醒词阈值调整为: {self.threshold}")

    @property
    def is_available(self) -> bool:
        """检查唤醒词检测是否可用"""
        return self._model is not None
