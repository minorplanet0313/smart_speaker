"""
状态机 — 管理智能音箱交互状态

状态转换:
    IDLE → LISTENING (wake word)
    LISTENING → THINKING (speech end)
    LISTENING → IDLE (timeout)
    THINKING → SPEAKING (TTS ready)
    THINKING → IDLE (empty ASR / error)
    SPEAKING → IDLE (playback done)
    SPEAKING → LISTENING (barge-in)
    ANY → MUTED (button)
    MUTED → IDLE (button)
    ANY → ERROR (critical error)
    ERROR → IDLE (recovery)
"""

import threading
import time
from enum import Enum
from typing import Callable, Dict, List, Optional, Set, Tuple

from src.core.event_bus import Event, EventBus
from src.utils.logger import get_logger

logger = get_logger(__name__)


class State(Enum):
    """交互状态"""
    IDLE = "idle"              # 等待唤醒词
    LISTENING = "listening"    # 正在听用户说话
    THINKING = "thinking"      # 处理中 (ASR → LLM → TTS)
    SPEAKING = "speaking"      # 正在播放回复
    MUTED = "muted"            # 静音模式
    ERROR = "error"            # 错误状态


# 允许的状态转换: from_state → {to_states}
STATE_TRANSITIONS: Dict[State, Set[State]] = {
    State.IDLE:      {State.LISTENING, State.MUTED, State.ERROR},
    State.LISTENING: {State.IDLE, State.THINKING, State.MUTED, State.ERROR},
    State.THINKING:  {State.IDLE, State.SPEAKING, State.MUTED, State.ERROR},
    State.SPEAKING:  {State.IDLE, State.LISTENING, State.MUTED, State.ERROR},
    State.MUTED:     {State.IDLE, State.ERROR},
    State.ERROR:     {State.IDLE, State.ERROR},  # ERROR→ERROR 用于重试
}


class StateMachine:
    """
    状态机

    特性:
    - 严格的状态转换检查
    - 状态变更通知 (通过 EventBus)
    - 状态超时保护
    - 线程安全
    """

    def __init__(self, event_bus: Optional[EventBus] = None):
        self._state = State.IDLE
        self._lock = threading.RLock()
        self._event_bus = event_bus or EventBus.instance()
        self._state_enter_time: Dict[State, float] = {}
        self._on_change_callbacks: List[Callable[[State, State], None]] = []
        self._timeout_callbacks: Dict[State, Tuple[float, Callable]] = {}

    @property
    def current_state(self) -> State:
        return self._state

    @property
    def state_duration_ms(self) -> float:
        """当前状态的持续时间 (毫秒)"""
        enter_time = self._state_enter_time.get(self._state, time.time())
        return (time.time() - enter_time) * 1000

    def transition(self, to_state: State) -> bool:
        """
        尝试转换到目标状态

        Returns:
            True 如果转换成功, False 如果转换被拒绝
        """
        with self._lock:
            from_state = self._state

            if to_state == from_state:
                return True

            allowed = STATE_TRANSITIONS.get(from_state, set())
            if to_state not in allowed:
                logger.warning(
                    f"状态转换被拒绝: {from_state.value} → {to_state.value} "
                    f"(允许: {[s.value for s in allowed]})"
                )
                return False

            # 执行转换
            self._state = to_state
            self._state_enter_time[to_state] = time.time()

            logger.info(
                f"状态转换: {from_state.value} → {to_state.value} "
                f"(在 {from_state.value} 中停留了 {self.state_duration_ms:.0f}ms)"
            )

        # 通过 EventBus 通知
        self._event_bus.publish(
            Event.STATE_CHANGED,
            source="state_machine",
            from_state=from_state.value,
            to_state=to_state.value,
        )

        # 调用注册的回调
        for callback in self._on_change_callbacks:
            try:
                callback(from_state, to_state)
            except Exception as e:
                logger.error(f"状态变更回调异常: {e}")

        return True

    def can_transition(self, to_state: State) -> bool:
        """检查是否允许转换到目标状态"""
        allowed = STATE_TRANSITIONS.get(self._state, set())
        return to_state in allowed

    def is_in(self, *states: State) -> bool:
        """检查当前是否在指定状态中"""
        return self._state in states

    def on_change(self, callback: Callable[[State, State], None]) -> None:
        """注册状态变更回调"""
        self._on_change_callbacks.append(callback)

    def set_timeout(self, state: State, timeout_ms: float,
                    callback: Callable[[], None]) -> None:
        """
        为特定状态设置超时回调

        由 Engine 定时检查, 当状态停留时间超过 timeout_ms 时触发
        """
        self._timeout_callbacks[state] = (timeout_ms, callback)

    def check_timeouts(self) -> None:
        """检查并触发超时回调 (由 Engine 主循环调用)"""
        for state, (timeout_ms, callback) in list(self._timeout_callbacks.items()):
            if self._state == state:
                enter_time = self._state_enter_time.get(state, time.time())
                elapsed_ms = (time.time() - enter_time) * 1000
                if elapsed_ms > timeout_ms:
                    logger.warning(
                        f"状态超时: {state.value}, "
                        f"elapsed={elapsed_ms:.0f}ms > timeout={timeout_ms:.0f}ms"
                    )
                    try:
                        callback()
                    except Exception as e:
                        logger.error(f"超时回调异常: {e}")
                    # 移除一次性超时
                    self._timeout_callbacks.pop(state, None)

    def force_idle(self) -> None:
        """
        强制回到 IDLE 状态 (异常恢复)
        不论当前什么状态, 直接切到 IDLE
        """
        with self._lock:
            old_state = self._state
            self._state = State.IDLE
            self._state_enter_time[State.IDLE] = time.time()

        logger.warning(f"强制状态重置: {old_state.value} → IDLE")

        self._event_bus.publish(
            Event.STATE_CHANGED,
            source="state_machine",
            from_state=old_state.value,
            to_state=State.IDLE.value,
        )

    def reset(self) -> None:
        """重置状态机"""
        self.force_idle()
        self._timeout_callbacks.clear()
