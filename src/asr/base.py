"""
ASR (语音识别) 抽象接口

所有 ASR 实现 (本地 Vosk, 云端 API) 必须实现此接口
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class ASRResult:
    """ASR 识别结果"""
    text: str                          # 识别的文本
    confidence: float = 0.0            # 置信度 [0, 1]
    is_final: bool = True              # 是否最终结果 (流式识别时使用)
    latency_ms: float = 0.0            # 识别耗时 (毫秒)
    partial_text: str = ""             # 部分结果 (流式)
    metadata: dict = field(default_factory=dict)


class BaseASR(ABC):
    """ASR 抽象接口"""

    @abstractmethod
    def transcribe(
        self,
        audio_data: np.ndarray,
        sample_rate: int = 16000,
    ) -> ASRResult:
        """
        转写音频数据为文本

        Args:
            audio_data: 音频数据 (float32 数组)
            sample_rate: 采样率

        Returns:
            ASRResult 识别结果
        """
        ...

    @abstractmethod
    def transcribe_file(self, file_path: str) -> ASRResult:
        """转写音频文件"""
        ...

    @property
    @abstractmethod
    def is_available(self) -> bool:
        """检查 ASR 服务是否可用"""
        ...

    @abstractmethod
    def release(self) -> None:
        """释放资源"""
        ...
