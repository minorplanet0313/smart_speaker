"""
LLM (大语言模型) 抽象接口

所有 LLM 实现必须实现此接口
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, Iterator, List, Optional


@dataclass
class LLMResponse:
    """LLM 响应"""
    text: str
    conversation_id: str = ""
    tokens_used: int = 0
    latency_ms: float = 0.0
    finish_reason: str = "stop"  # stop | length | error
    metadata: dict = field(default_factory=dict)


@dataclass
class LLMMessage:
    """对话消息"""
    role: str       # "system" | "user" | "assistant"
    content: str


class BaseLLM(ABC):
    """LLM 抽象接口"""

    @abstractmethod
    def chat(
        self,
        messages: List[Dict[str, str]],
        stream: bool = False,
        **kwargs,
    ) -> LLMResponse:
        """
        发送消息并获取回复

        Args:
            messages: 对话消息列表
            stream: 是否流式返回

        Returns:
            LLMResponse
        """
        ...

    @abstractmethod
    def chat_stream(
        self,
        messages: List[Dict[str, str]],
        **kwargs,
    ) -> Iterator[str]:
        """
        流式聊天, 返回文本块迭代器
        """
        ...

    @property
    @abstractmethod
    def is_available(self) -> bool:
        """检查 LLM 服务是否可用"""
        ...
