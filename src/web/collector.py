"""
Web 事件收集器

订阅 EventBus 关键事件，维护环形历史缓冲区 + 当前状态快照。
所有订阅者回调极轻（只做 dict 组装 + queue.put），不阻塞音频管线。
"""

import collections
import queue
import threading
import time
from typing import Optional

from src.core.event_bus import Event, EventBus


class WebEventCollector:
    """轻量级事件收集器，为 Web UI 提供数据源"""

    def __init__(
        self,
        event_bus: EventBus,
        engine=None,
        max_history: int = 500,
    ):
        self._bus = event_bus
        self._engine = engine  # 用于 get_status() / get_messages()
        self._queue: queue.Queue = queue.Queue(maxsize=200)
        self._history = collections.deque(maxlen=max_history)
        self._lock = threading.Lock()

        # 当前状态快照
        self._state_snapshot: dict = {
            "state": "idle",
            "state_duration_ms": 0,
            "audio_rms": 0.0,
            "last_asr_text": "",
            "last_asr_confidence": 0.0,
            "last_wake_confidence": 0.0,
            "last_llm_response": "",
            "error_count": 0,
            "total_interactions": 0,
        }

        self._subscribe()

    # ---- 订阅 ----

    def _subscribe(self) -> None:
        events = [
            Event.STATE_CHANGED,
            Event.WAKE_WORD_DETECTED,
            Event.ASR_RESULT,
            Event.ASR_PARTIAL,
            Event.LLM_RESPONSE,
            Event.LLM_STREAM_CHUNK,
            Event.TTS_AUDIO_READY,
            Event.PLAYBACK_START,
            Event.PLAYBACK_DONE,
            Event.SPEECH_START,
            Event.SPEECH_END,
            Event.ERROR,
        ]
        for evt in events:
            self._bus.subscribe(evt, self._on_event)

    def _on_event(self, event_data) -> None:
        """EventBus 回调 — 必须极快返回，不阻塞"""
        data = dict(event_data.data)
        record = {
            "event": event_data.event.name,
            "source": event_data.source,
            "ts": time.time(),
            **data,
        }

        # 更新状态快照
        with self._lock:
            self._history.append(record)
            self._update_snapshot(event_data.event, data)

        # 推入实时队列
        try:
            self._queue.put_nowait(record)
        except queue.Full:
            pass

    def _update_snapshot(self, event: Event, data: dict) -> None:
        s = self._state_snapshot
        if event == Event.STATE_CHANGED:
            s["state"] = data.get("to", s["state"])
            s["state_duration_ms"] = 0
        elif event == Event.WAKE_WORD_DETECTED:
            s["last_wake_confidence"] = data.get("confidence", 0)
            s["total_interactions"] += 1
        elif event == Event.ASR_RESULT:
            s["last_asr_text"] = data.get("text", "")
            s["last_asr_confidence"] = data.get("confidence", 0)
        elif event == Event.ASR_PARTIAL:
            s["last_asr_text"] = "[识别中] " + data.get("partial_text", "")
        elif event == Event.LLM_RESPONSE:
            s["last_llm_response"] = data.get("text", "")[:200]
        elif event == Event.ERROR:
            s["error_count"] += 1

    # ---- 公共接口 ----

    def get_state(self) -> dict:
        """获取当前状态快照"""
        with self._lock:
            s = dict(self._state_snapshot)
        # 补充引擎实时状态
        if self._engine:
            eng = self._engine
            s["engine_state"] = eng.state_machine.current_state.value
            if hasattr(eng, '_audio_level_rms_sum') and hasattr(eng, '_audio_level_samples'):
                samples = eng._audio_level_samples
                if samples > 0:
                    s["audio_rms"] = round(eng._audio_level_rms_sum / samples, 4)
        return s

    def get_history(self, n: int = 50) -> list:
        """获取最近 N 条事件"""
        with self._lock:
            items = list(self._history)[-n:]
        return items

    def get_messages(self, n: int = 20) -> list:
        """获取最近对话消息"""
        messages = []
        if self._engine and self._engine.conversation_context:
            conv = self._engine.conversation_context
            try:
                c = conv._conversations.get(conv.current_conversation_id)
                if c:
                    messages = c.messages[-n:]
            except Exception:
                pass
        return messages

    def get_system_info(self) -> dict:
        """获取系统信息"""
        info = {}
        try:
            import psutil
            info["cpu_percent"] = psutil.cpu_percent(interval=0.1)
            info["memory_percent"] = psutil.virtual_memory().percent
            info["memory_used_mb"] = round(psutil.virtual_memory().used / 1024 / 1024, 1)
            info["memory_total_mb"] = round(psutil.virtual_memory().total / 1024 / 1024, 1)
            info["disk_percent"] = psutil.disk_usage("/").percent
            info["disk_free_gb"] = round(psutil.disk_usage("/").free / 1024 / 1024 / 1024, 1)
        except ImportError:
            info["psutil"] = False
        return info

    def get_config(self) -> dict:
        """获取当前配置（脱敏）"""
        if self._engine:
            raw = getattr(self._engine.config, '_data', {})
            # 脱敏：隐藏 api_key / secret 类字段
            safe = {}
            for k, v in raw.items():
                if isinstance(v, dict):
                    safe[k] = {
                        sk: ("***" if any(x in sk.lower() for x in ("key", "secret", "token")) else sv)
                        for sk, sv in v.items()
                    }
                else:
                    safe[k] = v
            return safe
        return {}

    def update_config(self, updates: dict) -> dict:
        """在线更新配置（写入 config.yaml）"""
        if not self._engine:
            return {"ok": False, "error": "引擎未初始化"}
        try:
            import yaml
            from pathlib import Path
            config_path = Path(self._engine.config._config_path)
            with open(config_path, 'r') as f:
                current = yaml.safe_load(f) or {}
            # 深度合并
            _deep_merge(current, updates)
            with open(config_path, 'w') as f:
                yaml.dump(current, f, allow_unicode=True, default_flow_style=False)
            return {"ok": True}
        except Exception as e:
            return {"ok": False, "error": str(e)}

    # ---- 实时流 ----

    def poll_events(self, timeout: float = 1.0) -> Optional[dict]:
        """阻塞等待下一个事件（供 SSE 使用）"""
        try:
            return self._queue.get(timeout=timeout)
        except queue.Empty:
            return None


def _deep_merge(base: dict, updates: dict) -> None:
    """递归合并 updates 到 base"""
    for k, v in updates.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _deep_merge(base[k], v)
        else:
            base[k] = v
