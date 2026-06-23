"""
对话上下文管理

维护多轮对话历史
- 限制最大对话轮数
- 上下文过期清理
- 支持多会话 (不同用户/会话ID)
"""

import time
import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class Conversation:
    """单个对话会话"""
    id: str
    messages: List[Dict[str, str]] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    last_active: float = field(default_factory=time.time)

    def add_user_message(self, text: str) -> None:
        self.messages.append({"role": "user", "content": text})
        self.last_active = time.time()

    def add_assistant_message(self, text: str) -> None:
        self.messages.append({"role": "assistant", "content": text})
        self.last_active = time.time()

    def get_messages(self, system_prompt: str = "") -> List[Dict[str, str]]:
        """获取消息列表 (包含 system prompt)"""
        result = []
        if system_prompt:
            result.append({"role": "system", "content": system_prompt})
        result.extend(self.messages)
        return result

    @property
    def rounds(self) -> int:
        """对话轮数 (一问一答算一轮)"""
        return len(self.messages) // 2

    @property
    def age_seconds(self) -> float:
        """会话存活时间"""
        return time.time() - self.created_at


class ConversationContext:
    """
    对话上下文管理器

    特性:
    - 自动生成会话 ID
    - 限制最大对话轮数 (超出自动截断)
    - 过期会话自动清理
    - 线程安全
    """

    def __init__(
        self,
        max_history_rounds: int = 20,
        context_timeout_seconds: int = 300,
        system_prompt: str = "",
    ):
        self.max_history_rounds = max_history_rounds
        self.context_timeout_seconds = context_timeout_seconds
        self.system_prompt = system_prompt

        self._conversations: Dict[str, Conversation] = {}
        self._current_conversation_id: Optional[str] = None

    @property
    def current_conversation_id(self) -> str:
        """获取或创建当前会话 ID"""
        if self._current_conversation_id is None:
            self.new_conversation()
        return self._current_conversation_id

    def new_conversation(self) -> str:
        """创建新会话"""
        conv_id = uuid.uuid4().hex[:12]
        self._conversations[conv_id] = Conversation(id=conv_id)
        self._current_conversation_id = conv_id
        self._cleanup_expired()
        logger.debug(f"新建会话: {conv_id}")
        return conv_id

    def add_user_message(self, text: str, conversation_id: str = None) -> None:
        """添加用户消息"""
        conv = self._get_conversation(conversation_id)
        conv.add_user_message(text)

    def add_assistant_message(self, text: str, conversation_id: str = None) -> None:
        """添加助手消息"""
        conv = self._get_conversation(conversation_id)
        conv.add_assistant_message(text)
        self._trim_if_needed(conv)

    def get_messages(self, conversation_id: str = None) -> List[Dict[str, str]]:
        """获取对话历史"""
        conv = self._get_conversation(conversation_id)
        return conv.get_messages(self.system_prompt)

    def clear(self, conversation_id: str = None) -> None:
        """清除会话历史"""
        conv_id = conversation_id or self.current_conversation_id
        if conv_id in self._conversations:
            self._conversations[conv_id].messages.clear()
            logger.debug(f"清除会话: {conv_id}")

    def _get_conversation(self, conversation_id: str = None) -> Conversation:
        """获取会话 (不存在则创建)"""
        conv_id = conversation_id or self.current_conversation_id
        if conv_id not in self._conversations:
            self._conversations[conv_id] = Conversation(id=conv_id)
        return self._conversations[conv_id]

    def _trim_if_needed(self, conv: Conversation) -> None:
        """如果对话轮数超出限制, 截断最早的对话 (保持配对)"""
        max_messages = self.max_history_rounds * 2  # 一问一答 = 2条消息
        if len(conv.messages) > max_messages:
            excess = len(conv.messages) - max_messages
            # 确保删除偶数条, 避免 "提问-回答" 配对被打散
            if excess % 2 != 0:
                excess += 1
            conv.messages = conv.messages[excess:]
            logger.debug(f"对话截断: 删除了 {excess} 条早期消息")

    def _cleanup_expired(self) -> None:
        """清理过期会话"""
        now = time.time()
        expired = [
            conv_id for conv_id, conv in self._conversations.items()
            if now - conv.last_active > self.context_timeout_seconds
        ]
        for conv_id in expired:
            del self._conversations[conv_id]
            logger.debug(f"清理过期会话: {conv_id}")

    @property
    def active_conversations(self) -> int:
        """活跃会话数量"""
        self._cleanup_expired()
        return len(self._conversations)
