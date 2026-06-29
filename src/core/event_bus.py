"""
事件总线 — 发布/订阅模式

用于模块间解耦通信, 所有事件通过 EventBus 传递。
每个事件类型可以有多个订阅者, 订阅者的执行是异步且串行的 (按注册顺序)。
"""

import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional

from src.utils.logger import get_logger

logger = get_logger(__name__)


class Event(Enum):
    """标准事件类型"""
    # 音频事件
    AUDIO_FRAME = auto()          # 原始音频帧
    AUDIO_DEVICE_ERROR = auto()   # 音频设备异常

    # 唤醒词事件
    WAKE_WORD_DETECTED = auto()   # 唤醒词检测到

    # VAD 事件
    SPEECH_START = auto()         # 语音开始
    SPEECH_END = auto()           # 语音结束 (携带音频数据)
    SPEECH_TIMEOUT = auto()       # 语音超时

    # ASR 事件
    ASR_RESULT = auto()           # 识别结果 (最终)
    ASR_PARTIAL = auto()          # 识别中间结果 (流式)
    ASR_ERROR = auto()            # 识别失败

    # LLM 事件
    LLM_RESPONSE = auto()         # LLM 完整响应
    LLM_STREAM_CHUNK = auto()     # LLM 流式输出块
    LLM_ERROR = auto()            # LLM 错误

    # TTS 事件
    TTS_AUDIO_READY = auto()      # TTS 音频数据就绪
    TTS_STREAM_CHUNK = auto()     # TTS 流式音频块
    TTS_DONE = auto()             # TTS 完成
    TTS_ERROR = auto()            # TTS 失败

    # 播放事件
    PLAYBACK_START = auto()       # 开始播放
    PLAYBACK_DONE = auto()        # 播放结束
    PLAYBACK_INTERRUPTED = auto() # 播放被打断

    # 状态事件
    STATE_CHANGED = auto()        # 状态机状态变更

    # 系统事件
    NETWORK_ONLINE = auto()       # 网络恢复
    NETWORK_OFFLINE = auto()      # 网络断开
    ERROR = auto()                # 通用错误
    BUTTON_PRESSED = auto()       # 物理按键按下
    SHUTDOWN = auto()             # 系统关闭

    # 技能事件
    SKILL_RESULT = auto()         # 技能执行结果

    # 配置事件
    CONFIG_UPDATED = auto()       # 配置在线更新


@dataclass
class EventData:
    """事件数据封装"""
    event: Event
    data: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    source: str = ""  # 事件来源模块名

    def get(self, key: str, default: Any = None) -> Any:
        return self.data.get(key, default)

    def __str__(self) -> str:
        return f"Event({self.event.name}, source={self.source}, data_keys={list(self.data.keys())})"


# 回调函数签名: Callable[[EventData], None]
EventCallback = Callable[[EventData], None]


class EventBus:
    """
    事件总线

    使用方式:
        bus = EventBus()

        # 订阅
        bus.subscribe(Event.WAKE_WORD_DETECTED, my_handler)

        # 发布
        bus.publish(Event.WAKE_WORD_DETECTED, confidence=0.95)

        # 取消订阅
        bus.unsubscribe(Event.WAKE_WORD_DETECTED, my_handler)
    """

    _instance: Optional["EventBus"] = None

    def __init__(self):
        self._subscribers: Dict[Event, List[EventCallback]] = defaultdict(list)
        self._lock = threading.RLock()

    @classmethod
    def instance(cls) -> "EventBus":
        """获取全局单例"""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def subscribe(self, event: Event, callback: EventCallback) -> None:
        """订阅事件"""
        with self._lock:
            if callback not in self._subscribers[event]:
                self._subscribers[event].append(callback)
                logger.debug(f"订阅事件: {event.name} <- {callback.__name__}")

    def unsubscribe(self, event: Event, callback: EventCallback) -> None:
        """取消订阅"""
        with self._lock:
            if callback in self._subscribers[event]:
                self._subscribers[event].remove(callback)
                logger.debug(f"取消订阅: {event.name} <- {callback.__name__}")

    def publish(self, event: Event, source: str = "", **data) -> None:
        """
        发布事件 (非阻塞, 每个订阅者同步调用)

        Args:
            event: 事件类型
            source: 事件来源 (模块名)
            **data: 事件携带的数据
        """
        event_data = EventData(event=event, data=data, source=source)

        with self._lock:
            subscribers = list(self._subscribers.get(event, []))

        if not subscribers:
            logger.debug(f"事件无订阅者: {event.name}")
            return

        for callback in subscribers:
            try:
                callback(event_data)
            except Exception as e:
                logger.error(
                    f"事件处理异常: event={event.name}, "
                    f"handler={callback.__name__}, error={e}",
                    exc_info=True
                )

    def clear(self) -> None:
        """清除所有订阅 (用于测试)"""
        with self._lock:
            self._subscribers.clear()

    @property
    def subscriber_count(self) -> Dict[Event, int]:
        """获取各事件订阅数"""
        with self._lock:
            return {e: len(subs) for e, subs in self._subscribers.items()}
