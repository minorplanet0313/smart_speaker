"""
TTS (语音合成) 抽象接口

所有 TTS 实现 (Edge TTS, Piper 等) 必须实现此接口
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Iterator, Optional


@dataclass
class TTSResult:
    """TTS 合成结果"""
    audio_data: bytes           # 音频二进制数据
    format: str = "mp3"         # 音频格式 (mp3, wav, pcm)
    sample_rate: int = 16000    # 采样率
    latency_ms: float = 0.0     # 合成耗时 (毫秒)
    metadata: dict = field(default_factory=dict)


class BaseTTS(ABC):
    """TTS 抽象接口"""

    @abstractmethod
    def synthesize(self, text: str) -> TTSResult:
        """
        将文本合成为语音

        Args:
            text: 要合成的文本

        Returns:
            TTSResult 包含音频数据

        Raises:
            RuntimeError: 合成失败
        """
        ...

    @abstractmethod
    def synthesize_stream(self, text: str) -> Iterator[bytes]:
        """
        流式语音合成

        Yields:
            音频块 (bytes)
        """
        ...

    @property
    @abstractmethod
    def is_available(self) -> bool:
        """检查 TTS 服务是否可用"""
        ...

    @property
    @abstractmethod
    def voice_name(self) -> str:
        """当前使用的语音名称"""
        ...
