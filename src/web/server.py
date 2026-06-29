"""
Bottle Web 服务器

轻量 HTTP + SSE 实时推送，在独立 daemon 线程运行。
路由:
  GET  /                  → 仪表盘 HTML
  GET  /api/status        → 当前状态 JSON
  GET  /api/history?n=50  → 最近事件 JSON
  GET  /api/messages?n=20 → 对话消息 JSON
  GET  /api/system        → 系统信息 JSON
  GET  /api/config        → 当前配置 JSON
  GET  /api/events/stream → SSE 实时事件流
  POST /api/text          → 发送文字指令
  POST /api/config        → 在线更新配置
  POST /api/wake          → 远程唤醒
"""

import json
import os
import threading
import time

from src.utils.logger import get_logger

logger = get_logger(__name__)


class WebServer:
    """Bottle + SSE Web 管理面板"""

    def __init__(
        self,
        collector,
        engine,
        host: str = "0.0.0.0",
        port: int = 8080,
    ):
        self.collector = collector
        self.engine = engine
        self.host = host
        self.port = port
        self._thread: threading.Thread = None
        self._running = False

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(
            target=self._serve,
            daemon=True,
            name="web-server",
        )
        self._thread.start()
        logger.info(f"Web UI: http://{self.host}:{self.port}")

    def stop(self) -> None:
        self._running = False

    def _serve(self) -> None:
        """在 daemon 线程中运行 Bottle"""
        from bottle import Bottle, request, response, static_file

        app = Bottle()
        collector = self.collector
        engine = self.engine

        # ---- 静态文件 ----

        @app.route("/")
        def index():
            return static_file("index.html", root=_static_dir())

        @app.route("/static/<filename:path>")
        def serve_static(filename):
            return static_file(filename, root=_static_dir())

        # ---- REST API ----

        @app.get("/api/status")
        def api_status():
            response.content_type = "application/json"
            return json.dumps(collector.get_state(), ensure_ascii=False)

        @app.get("/api/history")
        def api_history():
            n = int(request.query.get("n", 50))
            response.content_type = "application/json"
            return json.dumps(collector.get_history(n), ensure_ascii=False)

        @app.get("/api/messages")
        def api_messages():
            n = int(request.query.get("n", 20))
            response.content_type = "application/json"
            return json.dumps(collector.get_messages(n), ensure_ascii=False)

        @app.get("/api/system")
        def api_system():
            response.content_type = "application/json"
            return json.dumps(collector.get_system_info(), ensure_ascii=False)

        @app.get("/api/config")
        def api_config():
            response.content_type = "application/json"
            return json.dumps(collector.get_config(), ensure_ascii=False)

        @app.get("/api/audio/devices")
        def api_audio_devices():
            response.content_type = "application/json"
            return json.dumps(collector.get_audio_devices(), ensure_ascii=False)

        @app.get("/api/config/schema")
        def api_config_schema():
            from src.web.config_schema import get_schema_list
            response.content_type = "application/json"
            return json.dumps(get_schema_list(), ensure_ascii=False)

        @app.get("/api/config/categories")
        def api_config_categories():
            from src.web.config_schema import get_categories
            response.content_type = "application/json"
            return json.dumps(get_categories(), ensure_ascii=False)

        @app.post("/api/config")
        def api_config_update():
            try:
                updates = request.json
                result = collector.update_config(updates)
                response.content_type = "application/json"
                return json.dumps(result, ensure_ascii=False)
            except Exception as e:
                response.status = 400
                return json.dumps({"ok": False, "error": str(e)})

        @app.post("/api/text")
        def api_text():
            """接收文字指令，模拟 ASR 结果"""
            try:
                data = request.json or {}
                text = data.get("text", "").strip()
                if not text:
                    response.status = 400
                    return json.dumps({"ok": False, "error": "text is empty"})
                # 在独立线程中处理，避免阻塞 HTTP
                threading.Thread(
                    target=_handle_text_input,
                    args=(engine, text),
                    daemon=True,
                ).start()
                return json.dumps({"ok": True})
            except Exception as e:
                response.status = 500
                return json.dumps({"ok": False, "error": str(e)})

        @app.post("/api/wake")
        def api_wake():
            """远程触发唤醒"""
            try:
                from src.core.event_bus import Event, EventBus
                bus = EventBus.instance()
                # 模拟唤醒词检测
                threading.Thread(
                    target=_trigger_wake,
                    args=(engine, bus),
                    daemon=True,
                ).start()
                return json.dumps({"ok": True})
            except Exception as e:
                response.status = 500
                return json.dumps({"ok": False, "error": str(e)})

        # ---- SSE 实时流 ----

        @app.get("/api/events/stream")
        def api_events_stream():
            """Server-Sent Events 实时事件推送"""
            response.content_type = "text/event-stream"
            response.set_header("Cache-Control", "no-cache")
            response.set_header("Connection", "keep-alive")
            response.set_header("X-Accel-Buffering", "no")

            def event_generator():
                # 发送初始状态
                state = collector.get_state()
                state["system"] = collector.get_system_info()
                yield f"data: {json.dumps({'type': 'state', 'data': state}, ensure_ascii=False)}\n\n"

                while self._running:
                    event = collector.poll_events(timeout=1.0)
                    if event:
                        yield f"data: {json.dumps({'type': 'event', 'data': event}, ensure_ascii=False)}\n\n"
                    else:
                        # 心跳保活
                        yield ": heartbeat\n\n"

            return event_generator()

        # ---- CORS (开发用) ----

        @app.hook("after_request")
        def enable_cors():
            response.set_header("Access-Control-Allow-Origin", "*")
            response.set_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
            response.set_header("Access-Control-Allow-Headers", "Content-Type")

        @app.route("/<path:path>", method="OPTIONS")
        @app.route("/", method="OPTIONS")
        def options_handler(path=""):
            response.set_header("Access-Control-Allow-Origin", "*")
            response.set_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
            response.set_header("Access-Control-Allow-Headers", "Content-Type")
            return ""

        # 启动 (使用静默 handler，避免每次 API 请求都打 stderr 日志)
        from wsgiref.simple_server import make_server, WSGIRequestHandler

        class QuietHandler(WSGIRequestHandler):
            def log_message(self, fmt, *args):
                pass  # 静默 HTTP 访问日志

        self._httpd = make_server(self.host, self.port, app,
                                  handler_class=QuietHandler)
        logger.info(f"Web 服务器已启动: http://{self.host}:{self.port}")
        try:
            self._httpd.serve_forever(poll_interval=0.5)
        except Exception as e:
            if self._running:
                logger.error(f"Web 服务器异常: {e}")


# ---- 辅助函数 ----

def _static_dir() -> str:
    return os.path.join(os.path.dirname(__file__), "static")


def _handle_text_input(engine, text: str) -> None:
    """处理来自 Web 的文字输入"""
    from src.core.event_bus import Event, EventBus
    from src.core.state_machine import State
    bus = EventBus.instance()
    try:
        # 如果在播放中，打断
        if engine.state_machine.current_state == State.SPEAKING and engine.audio_player:
            engine.audio_player.stop()
            bus.publish(Event.PLAYBACK_INTERRUPTED, source="web")

        # 切到 LISTENING → THINKING
        engine.state_machine.force_idle()
        engine.state_machine.transition(State.LISTENING)
        engine.state_machine.transition(State.THINKING)

        # 直接发送文字（跳过 ASR）
        bus.publish(Event.ASR_RESULT, source="web", text=text, confidence=1.0)
    except Exception as e:
        logger.error(f"Web 文字输入处理失败: {e}")


def _trigger_wake(engine, bus) -> None:
    """远程触发唤醒"""
    from src.core.event_bus import Event
    from src.core.state_machine import State
    try:
        state = engine.state_machine.current_state
        if state == State.SPEAKING and engine.audio_player:
            engine.audio_player.stop()
        engine.state_machine.force_idle()
        engine._begin_utterance()
        engine.state_machine.transition(State.LISTENING)
        bus.publish(Event.WAKE_WORD_DETECTED, source="web", confidence=1.0)
    except Exception as e:
        logger.error(f"远程唤醒失败: {e}")
