"""
语音活动检测 (VAD)

基于 Silero VAD 模型, ONNX Runtime 推理
轻量级: 模型约 2MB, 推理 <10ms/chunk @ Pi4

检测状态:
    SILENCE_START → SPEECH_START → IN_SPEECH → SPEECH_END → IN_SILENCE → ...
"""

import time
from enum import Enum, auto
from typing import Optional

import numpy as np

from src.audio.ring_buffer import AudioRingBuffer
from src.utils.logger import get_logger

logger = get_logger(__name__)


class VADState(Enum):
    """VAD 检测状态"""
    SILENCE_START = auto()   # 进入静音
    SPEECH_START = auto()    # 检测到语音开始
    IN_SPEECH = auto()       # 语音进行中
    SPEECH_END = auto()      # 语音结束
    IN_SILENCE = auto()      # 静音中


class VoiceActivityDetector:
    """
    Silero VAD 封装

    检测语音的起始和结束, 用于确定用户说话何时结束。

    使用滑动窗口算法:
    - 当连续 N 帧检测到语音 → SPEECH_START
    - 当连续 M 帧检测到静音 → SPEECH_END
    """

    def __init__(
        self,
        threshold: float = 0.5,
        min_speech_duration_ms: int = 250,
        min_silence_duration_ms: int = 800,
        speech_pad_ms: int = 200,
        sample_rate: int = 16000,
    ):
        self.threshold = threshold
        self.min_speech_duration_ms = min_speech_duration_ms
        self.min_silence_duration_ms = min_silence_duration_ms
        self.speech_pad_ms = speech_pad_ms
        self.sample_rate = sample_rate

        # 延迟加载模型
        self._model = None
        self._get_speech_timestamps = None

        # 运行时状态
        self._is_speech = False
        self._speech_start_time: Optional[float] = None
        self._silence_start_time: Optional[float] = None
        self._speech_buffer: list = []
        self._silero_buffer = AudioRingBuffer(max_samples=4096)
        self._post_speech_pad_frames = 0
        self._post_speech_pad_remaining = 0
        self._noise_floor = 0.0  # 自适应噪声底噪
        self._noise_samples = 0
        self._reset_state()

    def _lazy_load_model(self) -> None:
        """延迟加载 Silero VAD 模型"""
        if self._model is not None:
            return
        try:
            from silero_vad import load_silero_vad
            self._model = load_silero_vad(onnx=True)
            logger.info("Silero VAD 模型加载完成 (ONNX)")
        except ImportError:
            logger.warning("silero-vad 未安装, 使用能量检测降级")
            self._model = None
        except Exception as e:
            logger.error(f"Silero VAD 模型加载失败: {e}")
            self._model = None

    def _reset_state(self) -> None:
        """重置检测状态"""
        self._is_speech = False
        self._speech_start_time = None
        self._silence_start_time = None
        self._speech_detected = False
        # 清除 Silero 内部缓冲，避免跨语音段残留
        self._silero_buffer.clear()
        self._post_speech_pad_frames = 0
        self._post_speech_pad_remaining = 0

    def process(self, audio_frame: np.ndarray) -> VADState:
        """
        处理一个音频帧, 返回当前 VAD 状态

        Args:
            audio_frame: float32 数组, shape=(N,), range=[-1, 1]

        Returns:
            当前 VAD 状态
        """
        is_speech = self.is_speech(audio_frame)
        now = time.time()

        if is_speech and not self._speech_detected:
            # 语音开始
            if self._speech_start_time is None:
                self._speech_start_time = now

            speech_duration_ms = (now - self._speech_start_time) * 1000
            if speech_duration_ms >= self.min_speech_duration_ms:
                self._speech_detected = True
                self._is_speech = True
                self._silence_start_time = None
                chunk_samples = max(len(audio_frame), 1)
                self._post_speech_pad_frames = max(
                    1,
                    int(self.speech_pad_ms * self.sample_rate / 1000 / chunk_samples),
                )
                self._post_speech_pad_remaining = self._post_speech_pad_frames
                logger.debug(f"语音开始 (duration={speech_duration_ms:.0f}ms)")
                return VADState.SPEECH_START

        elif is_speech and self._speech_detected:
            self._silence_start_time = None
            self._post_speech_pad_remaining = self._post_speech_pad_frames
            return VADState.IN_SPEECH

        elif not is_speech and self._speech_detected:
            # 可能的语音结束 — 先应用 speech_pad 延迟判定
            if self._post_speech_pad_remaining > 0:
                self._post_speech_pad_remaining -= 1
                return VADState.IN_SPEECH

            if self._silence_start_time is None:
                self._silence_start_time = now

            silence_duration_ms = (now - self._silence_start_time) * 1000
            if silence_duration_ms >= self.min_silence_duration_ms:
                # 确认语音结束
                speech_duration_ms = (
                    (self._silence_start_time - self._speech_start_time)
                    if self._speech_start_time
                    else 0
                ) * 1000
                logger.debug(
                    f"语音结束 (speech={speech_duration_ms:.0f}ms, "
                    f"silence={silence_duration_ms:.0f}ms)"
                )
                self._reset_state()
                return VADState.SPEECH_END
            else:
                return VADState.IN_SILENCE

        elif not is_speech and not self._speech_detected:
            # 静音中
            self._speech_start_time = None
            return VADState.IN_SILENCE

        return VADState.IN_SILENCE

    def is_speech(self, audio_frame: np.ndarray) -> bool:
        """
        检测单帧是否包含语音

        Args:
            audio_frame: float32 数组, range=[-1, 1]

        Returns:
            True 如果检测到语音
        """
        self._lazy_load_model()

        if self._model is not None:
            return self._is_speech_silero(audio_frame)
        else:
            return self._is_speech_energy(audio_frame)

    def _is_speech_silero(self, audio_frame: np.ndarray) -> bool:
        """使用 Silero VAD 模型检测"""
        try:
            # silero-vad 5.x 要求精确 512 帧 @16kHz (或 256 @8kHz)
            # 累积帧到内部 buffer，达到 512 帧后推理
            self._silero_buffer.append(audio_frame)

            required_samples = 512 if self.sample_rate == 16000 else 256
            if len(self._silero_buffer) < required_samples:
                return self._is_speech_energy(audio_frame)

            chunk = self._silero_buffer.consume(required_samples)
            if chunk is None:
                return self._is_speech_energy(audio_frame)

            # silero-vad 5.x 需要 PyTorch tensor
            try:
                speech_prob = self._model(chunk, self.sample_rate).item()
            except AttributeError:
                import torch
                tensor = torch.from_numpy(chunk)
                speech_prob = self._model(tensor, self.sample_rate).item()
            return speech_prob > self.threshold
        except Exception as e:
            logger.error(f"Silero VAD 推理失败: {e}")
            return self._is_speech_energy(audio_frame)

    def _is_speech_energy(self, audio_frame: np.ndarray) -> bool:
        """
        能量检测降级方案
        计算 RMS 能量, 与自适应阈值比较

        在前 2 秒自动估算环境噪声底噪, 阈值 = 噪声底噪 × 3
        稳定运行后使用固定乘数, 防止阈值漂移
        """
        rms = np.sqrt(np.mean(np.square(audio_frame)))

        # 自适应噪声底噪估算（前 100 帧 ≈ 6.4 秒内, 取较低 RMS 作为噪声估计）
        if self._noise_samples < 100:
            self._noise_samples += 1
            if self._noise_floor == 0.0 or rms < self._noise_floor * 2:
                # 指数移动平均, 平滑噪声估计
                alpha = 0.05
                self._noise_floor = (1 - alpha) * self._noise_floor + alpha * rms
            energy_threshold = max(self._noise_floor * 3, 0.005)
        else:
            # 稳定运行: 使用已收敛的噪声底噪
            energy_threshold = max(self._noise_floor * 3, 0.005)

        return rms > energy_threshold

    def reset(self) -> None:
        """重置 VAD 状态"""
        self._reset_state()

    @property
    def is_speech_active(self) -> bool:
        """当前是否在语音中"""
        return self._is_speech
