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

        # 系统信息缓存 (避免频繁 psutil 阻塞)
        self._system_cache: dict = {}
        self._system_cache_ts: float = 0.0
        self._system_cache_ttl: float = 2.0  # 缓存 2 秒

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
        # 清除不可 JSON 序列化的值（如 TTS audio_data bytes）
        for key in list(data.keys()):
            if isinstance(data[key], bytes):
                data[key] = f"[binary: {len(data[key])} bytes]"
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
            s["state"] = data.get("to_state", s["state"])
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
        """获取系统信息（缓存 TTL 内直接返回，避免 psutil 阻塞 SSE 循环）"""
        now = time.time()
        if self._system_cache and (now - self._system_cache_ts) < self._system_cache_ttl:
            return self._system_cache
        info = {}
        try:
            import psutil
            info["cpu_percent"] = psutil.cpu_percent(interval=0)
            info["memory_percent"] = psutil.virtual_memory().percent
            info["memory_used_mb"] = round(psutil.virtual_memory().used / 1024 / 1024, 1)
            info["memory_total_mb"] = round(psutil.virtual_memory().total / 1024 / 1024, 1)
            info["disk_percent"] = psutil.disk_usage("/").percent
            info["disk_free_gb"] = round(psutil.disk_usage("/").free / 1024 / 1024 / 1024, 1)
        except ImportError:
            info["psutil"] = False
        self._system_cache = info
        self._system_cache_ts = now
        return info

    def get_audio_devices(self) -> dict:
        """获取可用音频设备列表（过滤虚拟设备，只保留真实硬件）"""
        devices = {"input": [], "output": []}
        if self._engine:
            if self._engine.audio_capture:
                devices["input"] = _filter_real_devices(
                    self._engine.audio_capture.input_devices
                )
            if self._engine.audio_player:
                devices["output"] = _filter_real_devices(
                    self._engine.audio_player.output_devices
                )
        return devices

    def get_config(self) -> dict:
        """获取当前配置（递归脱敏 + 返回完整嵌套结构）"""
        if self._engine:
            raw = getattr(self._engine.config, '_data', {})
            return _sanitize_config(raw)
        return {}

    def update_config(self, updates: dict) -> dict:
        """在线更新配置: 校验 → 写磁盘 → 更新内存 → 发事件"""
        if not self._engine:
            return {"ok": False, "error": "引擎未初始化"}

        # 1. 校验
        from src.web.config_schema import validate_updates
        errors = validate_updates(updates)
        if errors:
            return {"ok": False, "error": "校验失败:\n" + "\n".join(f"  • {e}" for e in errors)}

        try:
            import yaml
            from pathlib import Path

            config_path = Path(self._engine.config._config_path)

            # 2. 读取当前配置 → 深度合并 → 写入磁盘
            with open(config_path, 'r') as f:
                current = yaml.safe_load(f) or {}
            _deep_merge(current, updates)
            with open(config_path, 'w') as f:
                yaml.dump(current, f, allow_unicode=True, default_flow_style=False)

            # 3. 同步更新内存中的 Config 单例
            changed = _flatten_keys(updates)
            for key in changed:
                val = _get_nested(current, key)
                if val is not None:
                    self._engine.config.update_path(key, val)

            # 4. 发布事件通知引擎
            EventBus.instance().publish(
                Event.CONFIG_UPDATED,
                source="web",
                updates=updates,
                changed_keys=list(changed),
            )

            return {"ok": True, "changed_keys": list(changed)}
        except Exception as e:
            return {"ok": False, "error": str(e)}

    # ---- 实时流 ----

    def poll_events(self, timeout: float = 1.0) -> Optional[dict]:
        """阻塞等待下一个事件（供 SSE 使用）"""
        try:
            return self._queue.get(timeout=timeout)
        except queue.Empty:
            return None


def _filter_real_devices(devices: list) -> list:
    """过滤虚拟/内部 ALSA 设备，只保留真实可用设备"""
    # 明显的虚拟/内部设备名黑名单
    _VIRTUAL_PATTERNS = [
        "samplerate", "speexrate", "surround", "dmix", "dsnooze",
        "front", "rear", "center_lfe", "side", "iec958",
        "hdmi", "modem", "phoneline", "usb_stream", "oss",
        "upmix", "vdownmix", "null",
    ]
    # 常用别名设备 — 即使声道数高也保留
    _KEEP_DEVICES = {"default", "pulse", "sysdefault"}
    result = []
    for d in devices:
        name = d.get("name", "")
        channels = d.get("channels", 0)
        is_keep = name.strip() in _KEEP_DEVICES
        # 跳过明显虚拟设备（声道数 > 16 肯定不是物理设备, 常用别名除外）
        if not is_keep and channels > 16:
            continue
        # 跳过名字含虚拟关键词的（常用别名除外）
        if not is_keep and any(p in name.lower() for p in _VIRTUAL_PATTERNS):
            continue
        result.append(d)
    return result


def _sanitize_config(d: dict) -> dict:
    """递归脱敏配置 — 所有层级中 key 含 key/secret/token 的叶子值替换为 ***"""
    result = {}
    for k, v in d.items():
        if isinstance(v, dict):
            result[k] = _sanitize_config(v)
        else:
            if any(x in k.lower() for x in ("key", "secret", "token")):
                result[k] = "***"
            else:
                result[k] = v
    return result


def _deep_merge(base: dict, updates: dict) -> None:
    """递归合并 updates 到 base"""
    for k, v in updates.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _deep_merge(base[k], v)
        else:
            base[k] = v


def _flatten_keys(d: dict, prefix: str = "") -> set:
    """将嵌套 dict 展开为 dot-path key 集合, e.g. {'audio.vad.threshold', ...}"""
    keys = set()
    for k, v in d.items():
        full_key = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict):
            keys |= _flatten_keys(v, full_key)
        else:
            keys.add(full_key)
    return keys


def _get_nested(d: dict, key_path: str):
    """按 dot-path 从嵌套 dict 中取值"""
    parts = key_path.split(".")
    current = d
    for p in parts:
        if isinstance(current, dict):
            current = current.get(p)
        else:
            return None
    return current
