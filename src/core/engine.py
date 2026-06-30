"""
主引擎 — Smart Speaker 的核心协调器

职责:
1. 创建和初始化所有模块
2. 连接事件总线, 协调模块间数据流
3. 管理主循环和生命周期
4. 异常处理和优雅关闭
"""

import queue
import signal
import sys
import threading
import time
from typing import Optional

import numpy as np

from src.core.event_bus import Event, EventBus
from src.core.state_machine import State, StateMachine
from src.audio.capture import AudioCapture
from src.audio.player import AudioPlayer
from src.audio.vad import VADState, VoiceActivityDetector
from src.wake_word.detector import WakeWordDetector
from src.wake_word.porcupine_detector import PorcupineDetector
from src.asr.vosk_asr import VoskASR
from src.asr.cloud_asr import CloudASR
from src.audio.preprocessing import preprocess_pipeline
from src.llm.deepseek import DeepSeekLLM
from src.llm.context import ConversationContext
from src.tts.edge_tts import EdgeTTS
from src.tts.piper_tts import PiperTTS
from src.tts.base import BaseTTS
from src.skills.base import SkillContext, SkillResult
from src.skills.skill_manager import SkillManager
from src.skills.builtin.chat_skill import ChatSkill
from src.skills.builtin.time_skill import TimeSkill
from src.utils.config import Config, get_config
from src.utils.logger import get_logger, setup_logger, suppress_alsa_noise, restore_alsa_noise
from src.utils.messages import ERROR_GENERIC, ERROR_NOT_UNDERSTOOD
from src.utils.sentence_split import extract_complete_sentences

logger = get_logger(__name__)


class SmartSpeakerEngine:
    """
    智能音箱主引擎

    使用示例:
        engine = SmartSpeakerEngine("config/config.yaml")
        engine.setup()
        engine.run_forever()
    """

    def __init__(self, config_path: str = "config/config.yaml"):
        self.config = get_config(config_path)
        self.event_bus = EventBus.instance()
        self.state_machine = StateMachine(self.event_bus)

        # 初始化日志
        log_level = self.config.get("general.log_level", "INFO")
        log_dir = self.config.get("general.data_dir", "data") + "/logs"
        setup_logger(level=log_level, log_dir=log_dir)

        # 模块引用 (在 setup() 中初始化)
        self.audio_capture: Optional[AudioCapture] = None
        self.audio_player: Optional[AudioPlayer] = None
        self.vad: Optional[VoiceActivityDetector] = None
        self.wake_word_detector: Optional[WakeWordDetector] = None
        self.asr: Optional[VoskASR] = None
        self.asr_cloud: Optional[CloudASR] = None
        self.llm: Optional[DeepSeekLLM] = None
        self.tts_edge: Optional[EdgeTTS] = None
        self.tts_piper: Optional[PiperTTS] = None
        self.tts_primary: Optional[BaseTTS] = None
        self.tts_fallback: Optional[BaseTTS] = None
        self.conversation_context: Optional[ConversationContext] = None
        self.skill_manager: Optional[SkillManager] = None
        self.web_server = None

        # 运行时状态
        self._running = False
        self._speech_lock = threading.Lock()
        self._speech_audio_buffer: list = []
        self._wake_word_cooldown_until = 0.0
        self._asr_incremental_mode = True
        self._asr_preprocess = True
        self._asr_final_text = ""
        self._save_debug_audio_enabled = False
        self._llm_stream_enabled = False
        self._streaming_tts_active = False
        self._stream_playback_started = False

        # 音频 offload 队列 + worker
        buffer_seconds = self.config.get("performance.audio_buffer_seconds", 0.5)
        sample_rate = self.config.get("audio.sample_rate", 16000)
        chunk_size = self.config.get("audio.chunk_size", 1024)
        max_queue = max(4, int(buffer_seconds * sample_rate / chunk_size))
        self._audio_queue: queue.Queue = queue.Queue(maxsize=max_queue)
        self._audio_worker: Optional[threading.Thread] = None
        self._audio_worker_running = False

        # 事件处理器引用 (用于 stop 时取消订阅)
        self._event_handlers: list = []

        # 注册信号处理
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    # ================================================================
    # 初始化
    # ================================================================

    def setup(self) -> None:
        """初始化所有模块"""
        logger.info("=" * 50)
        logger.info("Smart Speaker 引擎初始化中...")
        logger.info("=" * 50)

        self._init_audio()
        self._init_vad()
        self._init_wake_word()
        self._init_asr()
        self._init_llm()
        self._init_tts()
        self._init_skills()
        self._init_conversation_context()
        self._init_web()
        self._wire_events()
        self._setup_state_timeouts()

        logger.info("所有模块初始化完成 ✓")

    def _init_audio(self) -> None:
        """初始化音频捕获和播放"""
        logger.info("初始化音频模块...")
        self.audio_capture = AudioCapture(
            sample_rate=self.config.get("audio.sample_rate", 16000),
            channels=self.config.get("audio.channels", 1),
            chunk_size=self.config.get("audio.chunk_size", 1024),
            device_name=self.config.get("audio.device.microphone"),
            on_error=lambda msg: self.event_bus.publish(
                Event.AUDIO_DEVICE_ERROR, source="audio_capture", error=msg
            ),
        )
        self.audio_player = AudioPlayer(
            device_name=self.config.get("audio.device.speaker"),
        )

    def _init_vad(self) -> None:
        """初始化语音活动检测"""
        vad_config = self.config.get("audio.vad", {})
        if vad_config.get("enabled", True):
            self.vad = VoiceActivityDetector(
                threshold=vad_config.get("threshold", 0.5),
                min_speech_duration_ms=vad_config.get("min_speech_duration_ms", 250),
                min_silence_duration_ms=vad_config.get("min_silence_duration_ms", 800),
                speech_pad_ms=vad_config.get("speech_pad_ms", 200),
            )
            logger.info("VAD (Silero) 初始化完成")
        else:
            logger.info("VAD 已禁用")

    def _init_wake_word(self) -> None:
        """初始化唤醒词检测 (openWakeWord 或 Porcupine)"""
        ww_config = self.config.get("wake_word", {})
        if not ww_config.get("enabled", True):
            return

        engine = ww_config.get("engine", "openwakeword")

        if engine == "porcupine":
            self.wake_word_detector = PorcupineDetector(
                access_key=ww_config.get("porcupine_access_key", ""),
                keyword=ww_config.get("porcupine_keyword", "porcupine"),
                keyword_path=ww_config.get("porcupine_keyword_path", ""),
                sensitivity=ww_config.get("threshold", 0.7),
                model_path=ww_config.get("porcupine_model_path", ""),
            )
            logger.info(f"唤醒词检测 (Porcupine) 初始化完成, "
                        f"keyword={ww_config.get('porcupine_keyword', 'porcupine')}")
        else:
            self.wake_word_detector = WakeWordDetector(
                model_path=ww_config.get("model_path", ""),
                threshold=ww_config.get("threshold", 0.5),
                inference_framework=ww_config.get("inference_framework", "onnx"),
            )
            logger.info(f"唤醒词检测 (openWakeWord) 初始化完成, "
                        f"threshold={ww_config.get('threshold', 0.5)}")

            # 预检: 确保模型能加载
            if not self.wake_word_detector.is_available:
                logger.error("唤醒词模型加载失败! 语音唤醒不可用，请检查模型路径和 openwakeword 安装")

    def _init_asr(self) -> None:
        """初始化语音识别 (本地主引擎 + 云端备份)"""
        asr_config = self.config.get("asr", {})

        # --- 本地 ASR: 按 primary 选择引擎 ---
        primary = asr_config.get("primary", "vosk")
        if primary == "sherpa":
            sherpa_config = asr_config.get("sherpa", {})
            from src.asr.sherpa_asr import SherpaASR
            self.asr = SherpaASR(
                model_dir=sherpa_config.get("model_path",
                    "./models/sherpa-onnx-streaming-zipformer-zh-14M"),
                sample_rate=sherpa_config.get("sample_rate", 16000),
                num_threads=sherpa_config.get("num_threads", 2),
                decoding_method=sherpa_config.get("decoding_method", "greedy_search"),
            )
            if self.asr.is_available:
                logger.info(f"ASR (sherpa-onnx) 初始化完成: "
                            f"{sherpa_config.get('model_path')}")
        else:
            # 默认 Vosk
            vosk_config = asr_config.get("vosk", {})
            self.asr = VoskASR(
                model_path=vosk_config.get("model_path",
                    "models/vosk-model-cn-0.22"),
                sample_rate=vosk_config.get("sample_rate", 16000),
            )
            if self.asr.is_available:
                logger.info(f"ASR (Vosk) 初始化完成: {vosk_config.get('model_path')}")
            else:
                logger.warning("ASR (Vosk) 模型未找到, 将使用云端 ASR 降级")

        # --- 云端 ASR 备份 (百度/腾讯/阿里云) ---
        cloud_config = asr_config.get("cloud", {})
        cloud_provider = cloud_config.get("provider", "baidu")
        if cloud_provider == "baidu":
            baidu_cfg = cloud_config.get("baidu", {})
            self.asr_cloud = CloudASR(
                provider="baidu",
                api_key=baidu_cfg.get("api_key", ""),
                secret_key=baidu_cfg.get("secret_key", ""),
            )
        elif cloud_provider == "tencent":
            tencent_cfg = cloud_config.get("tencent", {})
            self.asr_cloud = CloudASR(
                provider="tencent",
                api_key=tencent_cfg.get("secret_id", ""),
                secret_key=tencent_cfg.get("secret_key", ""),
                region=tencent_cfg.get("region", "ap-guangzhou"),
            )
        elif cloud_provider == "aliyun":
            ali_cfg = cloud_config.get("aliyun", {})
            self.asr_cloud = CloudASR(
                provider="aliyun",
                api_key=ali_cfg.get("access_key_id", ""),
                secret_key=ali_cfg.get("access_key_secret", ""),
            )

        if self.asr_cloud and self.asr_cloud.is_available:
            logger.info(f"云端 ASR ({cloud_provider}) 就绪, 作为本地 ASR 降级备份")
        else:
            logger.info("云端 ASR 未配置, 仅使用本地 ASR")

        # 读取 ASR 优化开关
        self._asr_incremental_mode = asr_config.get("incremental", True)
        self._asr_preprocess = asr_config.get("preprocess", True)
        logger.info(
            f"ASR 优化: 增量识别={'✓' if self._asr_incremental_mode else '✗'}, "
            f"预处理={'✓' if self._asr_preprocess else '✗'}"
        )

    def _init_llm(self) -> None:
        """初始化大语言模型"""
        llm_config = self.config.get("llm", {})
        retry_config = llm_config.get("retry", {})
        self._llm_stream_enabled = llm_config.get("stream", False)
        self.llm = DeepSeekLLM(
            api_key=llm_config.get("api_key", ""),
            model=llm_config.get("model", "deepseek-chat"),
            base_url=llm_config.get("base_url", "https://api.deepseek.com"),
            system_prompt=self.config.get("conversation.system_prompt", ""),
            temperature=llm_config.get("temperature", 0.7),
            max_tokens=llm_config.get("max_tokens", 1024),
            timeout=llm_config.get("timeout_seconds", 30),
            max_retries=retry_config.get("max_retries", 3),
            backoff_base=retry_config.get("backoff_base", 2),
        )
        logger.info(
            f"LLM (DeepSeek) 初始化完成: model={llm_config.get('model')}, "
            f"stream={'✓' if self._llm_stream_enabled else '✗'}"
        )

    def _init_tts(self) -> None:
        """初始化语音合成"""
        tts_config = self.config.get("tts", {})
        edge_config = tts_config.get("edge", {})
        self.tts_edge = EdgeTTS(
            voice=edge_config.get("voice", "zh-CN-XiaoxiaoNeural"),
            rate=edge_config.get("rate", "+0%"),
            pitch=edge_config.get("pitch", "+0Hz"),
        )
        logger.info(f"TTS (Edge) 初始化完成: voice={edge_config.get('voice')}")

        # Piper 作为离线备份
        piper_config = tts_config.get("piper", {})
        self.tts_piper = PiperTTS(
            model_path=piper_config.get("model_path", ""),
            config_path=piper_config.get("config_path"),
        )
        if self.tts_piper.is_available:
            logger.info("TTS (Piper) 离线备份就绪")
        else:
            logger.info("TTS (Piper) 未配置, 仅使用在线 TTS")

        primary = tts_config.get("primary", "edge")
        fallback = tts_config.get("fallback", "piper")
        self.tts_primary = self.tts_edge if primary == "edge" else self.tts_piper
        self.tts_fallback = self.tts_piper if fallback == "piper" else self.tts_edge
        logger.info(f"TTS 主引擎: {primary}, 降级: {fallback}")

    def _init_skills(self) -> None:
        """初始化技能管理器"""
        self.skill_manager = SkillManager()
        skills_config = self.config.get("skills", {})
        builtin_enabled = skills_config.get(
            "builtin_enabled", ["chat", "time", "weather"]
        )

        if "chat" in builtin_enabled:
            self.skill_manager.register(ChatSkill(self.llm))
        if "time" in builtin_enabled:
            self.skill_manager.register(TimeSkill())

        weather_api_key = skills_config.get("weather", {}).get("api_key", "")
        if "weather" in builtin_enabled and weather_api_key:
            from src.skills.builtin.weather_skill import WeatherSkill
            weather_cfg = skills_config.get("weather", {})
            default_city = weather_cfg.get("city", "auto")
            api_host = weather_cfg.get("api_host", "devapi.qweather.com")
            self.skill_manager.register(
                WeatherSkill(weather_api_key, default_city, api_host)
            )

        logger.info(
            f"技能管理器初始化完成: "
            f"{len(self.skill_manager.list_skills())} 个技能已注册"
        )

    def _init_conversation_context(self) -> None:
        """初始化对话上下文管理"""
        conv_config = self.config.get("conversation", {})
        self.conversation_context = ConversationContext(
            max_history_rounds=conv_config.get("max_history_rounds", 20),
            context_timeout_seconds=conv_config.get("context_timeout_seconds", 300),
            system_prompt=conv_config.get("system_prompt", ""),
        )

        self._save_debug_audio_enabled = self.config.get("debug.save_audio", False)
        logger.info(f"对话上下文管理初始化完成: "
                     f"max_rounds={conv_config.get('max_history_rounds', 20)}")

    def _init_web(self) -> None:
        """初始化 Web 管理面板 (可选)"""
        web_config = self.config.get("web", {})
        if not web_config.get("enabled", False):
            return
        from src.web.collector import WebEventCollector
        from src.web.server import WebServer
        self._web_collector = WebEventCollector(
            event_bus=self.event_bus,
            engine=self,
        )
        self.web_server = WebServer(
            collector=self._web_collector,
            engine=self,
            host=web_config.get("host", "0.0.0.0"),
            port=web_config.get("port", 8080),
        )
        self.web_server.start()
        logger.info(f"Web 管理面板: http://{web_config.get('host', '0.0.0.0')}:{web_config.get('port', 8080)}")

    # ================================================================
    # 事件连接
    # ================================================================

    def _wire_events(self) -> None:
        """连接模块间的事件"""
        logger.info("连接事件总线...")

        handlers = [
            (Event.SPEECH_END, self._on_speech_end),
            (Event.ASR_RESULT, self._on_asr_result),
            (Event.ASR_PARTIAL, self._on_asr_partial),
            (Event.LLM_STREAM_CHUNK, self._on_llm_stream_chunk),
            (Event.LLM_RESPONSE, self._on_llm_response),
            (Event.TTS_AUDIO_READY, self._on_tts_audio_ready),
            (Event.TTS_STREAM_CHUNK, self._on_tts_stream_chunk),
            (Event.PLAYBACK_DONE, self._on_playback_done),
            (Event.ASR_ERROR, self._on_error),
            (Event.LLM_ERROR, self._on_error),
            (Event.TTS_ERROR, self._on_error),
            (Event.AUDIO_DEVICE_ERROR, self._on_error),
            (Event.ERROR, self._on_error),
            (Event.SHUTDOWN, self._on_shutdown),
            (Event.CONFIG_UPDATED, self._on_config_updated),
        ]
        for event, handler in handlers:
            self.event_bus.subscribe(event, handler)
            self._event_handlers.append((event, handler))

        logger.info("事件连接完成")

        if self.wake_word_detector:
            self.wake_word_detector.on_detected(self._on_wake_word_detected)

    def _setup_state_timeouts(self) -> None:
        """设置状态超时保护"""
        # LISTENING 状态最多 15 秒
        max_speech_ms = self.config.get("audio.vad.max_speech_duration_ms", 15000)
        self.state_machine.set_timeout(
            State.LISTENING, max_speech_ms,
            lambda: self._handle_state_timeout(State.LISTENING),
        )
        # THINKING 状态最多 30 秒
        self.state_machine.set_timeout(
            State.THINKING, 30000,
            lambda: self._handle_state_timeout(State.THINKING),
        )
        # SPEAKING 状态最多 60 秒
        self.state_machine.set_timeout(
            State.SPEAKING, 60000,
            lambda: self._handle_state_timeout(State.SPEAKING),
        )

    def _handle_state_timeout(self, state: State) -> None:
        """处理状态超时"""
        logger.error(f"状态超时: {state.value}, 强制回到 IDLE")
        if state == State.SPEAKING and self.audio_player:
            self.audio_player.stop()
        self.state_machine.force_idle()

    # ================================================================
    # 事件处理器
    # ================================================================

    # ================================================================
    # 音频 offload worker
    # ================================================================

    def _start_audio_worker(self) -> None:
        """启动音频处理 worker 线程"""
        if self._audio_worker_running:
            return
        self._audio_worker_running = True
        self._audio_worker = threading.Thread(
            target=self._audio_worker_loop,
            daemon=True,
            name="audio-worker",
        )
        self._audio_worker.start()

    def _stop_audio_worker(self) -> None:
        """停止音频 worker"""
        self._audio_worker_running = False
        try:
            self._audio_queue.put_nowait(None)
        except queue.Full:
            pass
        if self._audio_worker:
            self._audio_worker.join(timeout=2)
            self._audio_worker = None

    def _audio_worker_loop(self) -> None:
        """从队列取帧并处理唤醒/VAD/ASR feed"""
        while self._audio_worker_running:
            try:
                frame = self._audio_queue.get(timeout=0.2)
            except queue.Empty:
                continue
            if frame is None:
                break
            try:
                self._process_audio_frame(frame)
            except Exception as e:
                logger.error(f"音频帧处理异常: {e}", exc_info=True)

    def _process_audio_frame(self, audio_frame: np.ndarray) -> None:
        """处理音频帧: 路由到唤醒词检测器 (IDLE) 或 VAD (LISTENING)"""
        current_state = self.state_machine.current_state

        # 音频电平采集 — 所有状态都记录，确保 Web UI 音频表实时刷新
        if not hasattr(self, '_audio_level_debug_time'):
            self._audio_level_debug_time = 0.0
            self._audio_level_samples = 0
            self._audio_level_rms_sum = 0.0
        self._audio_level_samples += 1
        self._audio_level_rms_sum += float(
            np.sqrt(np.mean(np.square(audio_frame)))
        )
        now = time.time()
        if now - self._audio_level_debug_time >= 5.0:
            avg_rms = self._audio_level_rms_sum / max(self._audio_level_samples, 1)
            logger.debug(
                f"🎤 音频电平: RMS={avg_rms:.4f} "
                f"({'正常' if avg_rms > 0.001 else '⚠️ 太弱/无声!'}) "
                f"[{self._audio_level_samples} 帧/5s]"
            )
            self._audio_level_debug_time = now
            self._audio_level_samples = 0
            self._audio_level_rms_sum = 0.0

        if current_state == State.IDLE and self.wake_word_detector:
            if now < self._wake_word_cooldown_until:
                return
            self.wake_word_detector.detect(audio_frame)

        elif current_state == State.LISTENING:
            keep_buffer = (
                self._save_debug_audio_enabled
                or not self._asr_incremental_mode
            )
            if keep_buffer:
                with self._speech_lock:
                    self._speech_audio_buffer.append(audio_frame.copy())

            if self._asr_incremental_mode and self.asr and self.asr.is_available:
                try:
                    def on_partial(partial_text: str):
                        self.event_bus.publish(
                            Event.ASR_PARTIAL,
                            source="asr",
                            partial_text=partial_text,
                        )

                    accepted = self.asr.feed_chunk(
                        audio_frame, partial_callback=on_partial
                    )
                    if accepted:
                        with self._speech_lock:
                            self._asr_final_text += accepted
                except Exception as e:
                    logger.warning(f"增量 ASR feed 失败: {e}")

            if self.vad:
                vad_state = self.vad.process(audio_frame)
                if vad_state == VADState.SPEECH_START:
                    self.event_bus.publish(Event.SPEECH_START, source="vad")
                elif vad_state == VADState.SPEECH_END:
                    self.event_bus.publish(Event.SPEECH_END, source="vad")

    def _begin_utterance(self) -> None:
        """开始新一轮语音采集 (唤醒词/键盘唤醒共用)"""
        with self._speech_lock:
            self._speech_audio_buffer = []
            self._asr_final_text = ""
        if self._asr_incremental_mode and self.asr and self.asr.is_available:
            try:
                self.asr.begin_utterance()
            except Exception as e:
                logger.warning(f"增量 ASR 初始化失败: {e}")

    def _on_wake_word_detected(self, confidence: float) -> None:
        """唤醒词检测到: 切换到 LISTENING 状态"""
        logger.info(f"🎤 唤醒! (confidence={confidence:.3f})")

        # 设置冷却时间
        cooldown_ms = self.config.get("wake_word.cooldown_ms", 2000)
        self._wake_word_cooldown_until = time.time() + cooldown_ms / 1000.0

        # 如果在播放中, 打断 (barge-in)
        if self.state_machine.current_state == State.SPEAKING and self.audio_player:
            logger.info("打断当前播放 (barge-in)")
            self.audio_player.stop()
            self.event_bus.publish(Event.PLAYBACK_INTERRUPTED, source="engine")

        # 清空之前的语音缓冲并初始化增量 ASR
        self._begin_utterance()

        # 切换到 LISTENING
        if self.state_machine.transition(State.LISTENING):
            self.event_bus.publish(
                Event.WAKE_WORD_DETECTED,
                source="wake_word",
                confidence=confidence,
            )

    def _on_speech_end(self, event_data) -> None:
        """语音结束: 轻量切换状态，预处理与 ASR 在 worker 线程执行"""
        if not self.state_machine.is_in(State.LISTENING):
            return

        with self._speech_lock:
            buffer_len = len(self._speech_audio_buffer)
            if buffer_len == 0 and not (
                self._asr_incremental_mode and self.asr and self.asr.is_available
            ):
                logger.info("语音缓冲为空, 回到 IDLE")
                self.state_machine.transition(State.IDLE)
                return
            audio_chunks = list(self._speech_audio_buffer)
            self._speech_audio_buffer = []

        logger.info(f"语音结束, 累积了 {buffer_len} 个音频块")
        self.state_machine.transition(State.THINKING)

        threading.Thread(
            target=self._run_asr_pipeline,
            args=(audio_chunks,),
            daemon=True,
            name="asr-thread",
        ).start()

    def _run_asr_pipeline(self, audio_chunks: list) -> None:
        """ASR 管线: 合并音频、预处理、识别 (在独立线程中)"""
        sample_rate = self.config.get("audio.sample_rate", 16000)

        if audio_chunks:
            audio_data = np.concatenate(audio_chunks)
        elif self._asr_incremental_mode:
            audio_data = np.array([], dtype=np.float32)
        else:
            logger.info("无音频数据, 回到 IDLE")
            self.state_machine.transition(State.IDLE)
            return

        if self._asr_preprocess and len(audio_data) > 0:
            audio_data = preprocess_pipeline(audio_data, sample_rate=sample_rate)

        if self._save_debug_audio_enabled and len(audio_data) > 0:
            self._save_debug_audio(audio_data)

        if self._asr_incremental_mode and self.asr and self.asr.is_available:
            self._do_asr_finalize(audio_data)
        else:
            self._do_asr(audio_data)

    def _do_asr(self, audio_data: np.ndarray) -> None:
        """执行语音识别 (在独立线程中, 支持流式部分结果, 失败时降级到云端)"""
        try:
            # 流式部分结果回调: 发布 ASR_PARTIAL 事件
            def on_partial(partial_text: str):
                self.event_bus.publish(
                    Event.ASR_PARTIAL,
                    source="asr",
                    partial_text=partial_text,
                )

            result = self.asr.transcribe(audio_data, 16000, partial_callback=on_partial)
            if result and result.text.strip():
                logger.info(
                    f"ASR 结果: \"{result.text}\" "
                    f"(置信度={result.confidence:.2f}, "
                    f"延迟={result.latency_ms:.0f}ms)"
                )
                self.event_bus.publish(
                    Event.ASR_RESULT,
                    source="asr",
                    text=result.text,
                    confidence=result.confidence,
                    latency_ms=result.latency_ms,
                )
            else:
                logger.info("ASR 结果为空")
                self.event_bus.publish(Event.ASR_RESULT, source="asr", text="")
        except Exception as e:
            logger.warning(f"本地 ASR 失败: {e}, 尝试云端 ASR...")
            self._do_cloud_asr(audio_data)

    def _do_cloud_asr(self, audio_data: np.ndarray) -> None:
        """云端 ASR 降级 (本地 Vosk 失败时自动切换)"""
        if not self.asr_cloud or not self.asr_cloud.is_available:
            logger.error("云端 ASR 不可用, ASR 失败")
            self.event_bus.publish(
                Event.ASR_ERROR, source="asr",
                error="本地和云端 ASR 均不可用"
            )
            return

        try:
            result = self.asr_cloud.transcribe(audio_data, 16000)
            if result and result.text.strip():
                logger.info(
                    f"云端 ASR 结果: \"{result.text}\" "
                    f"(延迟={result.latency_ms:.0f}ms)"
                )
                self.event_bus.publish(
                    Event.ASR_RESULT,
                    source="asr_cloud",
                    text=result.text,
                    confidence=result.confidence,
                    latency_ms=result.latency_ms,
                )
            else:
                logger.info("云端 ASR 结果为空")
                self.event_bus.publish(Event.ASR_RESULT, source="asr_cloud", text="")
        except Exception as e2:
            logger.error(f"云端 ASR 也失败: {e2}")
            self.event_bus.publish(
                Event.ASR_ERROR, source="asr_cloud", error=str(e2)
            )

    def _handle_user_text(self, text: str) -> None:
        """统一处理用户文本 (ASR 结果 / 键盘输入)"""
        context = SkillContext(
            conversation_id=self.conversation_context.current_conversation_id,
        )
        result = self.skill_manager.execute(text, context)

        if result and result.success and result.response_text:
            logger.info(f"技能直接响应: \"{result.response_text}\"")
            self._do_tts(result.response_text)
        elif result and not result.success:
            logger.error(f"技能执行失败: {result.error_message}")
            self._do_tts(ERROR_GENERIC)
        elif result and result.success and result.needs_llm:
            self._do_llm(text)
        else:
            logger.warning(f"未找到合适的技能处理: \"{text}\"")
            self._do_tts(ERROR_NOT_UNDERSTOOD)

    def _on_asr_result(self, event_data) -> None:
        """ASR 结果: 通过技能管理器处理"""
        text = event_data.get("text", "").strip()

        if not text:
            logger.info("ASR 返回空文本, 回到 IDLE")
            self.state_machine.transition(State.IDLE)
            return

        self._handle_user_text(text)

    def _do_asr_finalize(self, audio_data: np.ndarray) -> None:
        """
        增量模式 ASR 最终化 (在独立线程中).
        音频已在 LISTENING 期间逐帧送入, 这里只需获取最终结果.
        失败时依次降级: 批量 Vosk → 云端 ASR.
        """
        start_time = time.time()
        try:
            final_text = self.asr.end_utterance()
            with self._speech_lock:
                combined = (self._asr_final_text + final_text).strip()
            latency_ms = (time.time() - start_time) * 1000

            if combined:
                logger.info(
                    f"ASR 结果 (增量): \"{combined}\" "
                    f"(finalize={latency_ms:.0f}ms)"
                )
                self.event_bus.publish(
                    Event.ASR_RESULT,
                    source="asr",
                    text=combined,
                    confidence=0.85,
                    latency_ms=latency_ms,
                )
            else:
                logger.info("ASR 结果为空 (增量模式)")
                self.event_bus.publish(Event.ASR_RESULT, source="asr", text="")
        except Exception as e:
            logger.warning(f"增量 ASR 最终化失败: {e}")
            # 降级到批量 Vosk → 批量内部失败会再降级到云端
            logger.info("Vosk 增量失败, 降级到批量模式...")
            self._do_asr(audio_data)

    def _on_asr_partial(self, event_data) -> None:
        """ASR 流式部分结果: 实时显示中间识别文字"""
        partial_text = event_data.get("partial_text", "")
        if partial_text:
            # 实时打印中间结果, 让用户看到识别正在进行
            logger.info(f"🎤 识别中... \"{partial_text}\"")

    def _on_llm_stream_chunk(self, event_data) -> None:
        """LLM 流式输出块: 累积文本 (用于调试/显示)"""
        delta = event_data.get("delta", "")
        # 可以在这里添加屏幕显示或其他实时反馈
        # 注意: 完整的响应由 LLM_RESPONSE 事件携带

    def _on_llm_response(self, event_data) -> None:
        """LLM 完成响应: 非流式模式下触发 TTS"""
        text = event_data.get("text", "").strip()
        if text:
            logger.info(f"LLM 响应: \"{text[:100]}{'...' if len(text) > 100 else ''}\"")

        if event_data.get("streamed"):
            return  # 流式模式已在 _do_llm_streaming 中逐句播放 TTS

        if not text:
            logger.warning("LLM 返回空文本")
            self.state_machine.transition(State.IDLE)
            return

        self._do_tts(text)

    def _save_debug_audio(self, audio_data: np.ndarray) -> None:
        """保存录制音频到文件 (调试用, 需 debug.save_audio=true)"""
        if not self._save_debug_audio_enabled:
            return
        try:
            import wave
            from pathlib import Path
            debug_dir = Path(
                self.config.get("general.data_dir", "data") + "/debug_audio"
            )
            debug_dir.mkdir(parents=True, exist_ok=True)
            import time as _time
            filename = debug_dir / f"speech_{_time.strftime('%Y%m%d_%H%M%S')}.wav"
            audio_int16 = (np.clip(audio_data, -1.0, 1.0) * 32767).astype(np.int16)
            with wave.open(str(filename), 'wb') as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(16000)
                wf.writeframes(audio_int16.tobytes())
            logger.info(f"📝 录音已保存: {filename} "
                        f"({len(audio_data)/16000:.1f}s, RMS={np.sqrt(np.mean(np.square(audio_data))):.4f})")
        except Exception as e:
            logger.error(f"保存调试音频失败: {e}")

    def _do_llm(self, text: str) -> None:
        """执行 LLM 调用 (异步)"""
        threading.Thread(
            target=self._do_llm_sync,
            args=(text,),
            daemon=True,
            name="llm-thread",
        ).start()

    def _do_llm_sync(self, text: str) -> None:
        """同步执行 LLM 调用"""
        try:
            self.conversation_context.add_user_message(text)
            messages = self.conversation_context.get_messages()

            logger.info(f"调用 LLM: \"{text[:50]}...\"")

            if self._llm_stream_enabled:
                self._do_llm_streaming(messages)
                return

            full_response = ""
            for delta in self.llm.chat_stream(messages):
                full_response += delta

            if not full_response:
                logger.warning("LLM 返回空响应")
                self.state_machine.transition(State.IDLE)
                return

            self.conversation_context.add_assistant_message(full_response)
            self.event_bus.publish(
                Event.LLM_RESPONSE,
                source="llm",
                text=full_response,
            )
        except Exception as e:
            logger.error(f"LLM 调用失败: {e}")
            self.event_bus.publish(
                Event.LLM_ERROR, source="llm", error=str(e)
            )

    def _do_llm_streaming(self, messages) -> None:
        """流式 LLM → 分句 TTS → 流式播放"""
        full_response = ""
        sentence_buffer = ""
        self._streaming_tts_active = True
        self._stream_playback_started = False

        try:
            for delta in self.llm.chat_stream(messages):
                full_response += delta
                sentence_buffer += delta

                self.event_bus.publish(
                    Event.LLM_STREAM_CHUNK,
                    source="llm",
                    delta=delta,
                )

                sentences, sentence_buffer = extract_complete_sentences(
                    sentence_buffer
                )
                for sentence in sentences:
                    self._synthesize_and_stream_sentence(sentence)

            remainder = sentence_buffer.strip()
            if remainder:
                self._synthesize_and_stream_sentence(remainder)

            if self._stream_playback_started and self.audio_player:
                self.audio_player.end_stream()
                self.audio_player.wait_stream_done()

            if not full_response:
                logger.warning("LLM 返回空响应")
                self.state_machine.transition(State.IDLE)
                return

            self.conversation_context.add_assistant_message(full_response)
            logger.info(
                f"LLM 响应 (流式): \"{full_response[:100]}"
                f"{'...' if len(full_response) > 100 else ''}\""
            )
            self.event_bus.publish(
                Event.LLM_RESPONSE,
                source="llm",
                text=full_response,
                streamed=True,
            )

            if self._stream_playback_started:
                self.event_bus.publish(Event.PLAYBACK_DONE, source="player")
            else:
                self.state_machine.transition(State.IDLE)

        except Exception as e:
            logger.error(f"流式 LLM 失败: {e}")
            self.event_bus.publish(Event.LLM_ERROR, source="llm", error=str(e))
        finally:
            self._streaming_tts_active = False
            self._stream_playback_started = False

    def _synthesize_and_stream_sentence(self, text: str) -> None:
        """合成单句并发布流式 TTS 块"""
        if not text.strip():
            return
        try:
            tts = self.tts_primary
            if tts and tts.is_available:
                result = tts.synthesize(text)
                self.event_bus.publish(
                    Event.TTS_STREAM_CHUNK,
                    source="tts",
                    audio_data=result.audio_data,
                    format=result.format,
                    is_last=False,
                )
                return
        except Exception as e:
            logger.warning(f"主 TTS 失败: {e}")

        try:
            if self.tts_fallback and self.tts_fallback.is_available:
                if hasattr(self.tts_fallback, 'synthesize_stream'):
                    pcm_chunks = list(self.tts_fallback.synthesize_stream(text))
                    if pcm_chunks:
                        self.event_bus.publish(
                            Event.TTS_STREAM_CHUNK,
                            source="tts_piper",
                            audio_data=b"".join(pcm_chunks),
                            format="pcm",
                            sample_rate=22050,
                            is_last=False,
                        )
                        return
                result = self.tts_fallback.synthesize(text)
                self.event_bus.publish(
                    Event.TTS_STREAM_CHUNK,
                    source="tts_fallback",
                    audio_data=result.audio_data,
                    format=result.format,
                    is_last=False,
                )
        except Exception as e:
            logger.error(f"流式 TTS 失败: {e}")
            self.event_bus.publish(Event.TTS_ERROR, source="tts", error=str(e))

    def _do_tts(self, text: str) -> None:
        """执行 TTS 合成 (异步)"""
        threading.Thread(
            target=self._do_tts_sync,
            args=(text,),
            daemon=True,
            name="tts-thread",
        ).start()

    def _do_tts_sync(self, text: str) -> None:
        """同步执行 TTS, 尝试主 TTS → 离线备份"""
        try:
            result = self.tts_primary.synthesize(text)
            logger.info(
                f"TTS 完成: {result.latency_ms:.0f}ms, "
                f"{len(result.audio_data)} bytes"
            )
            self.event_bus.publish(
                Event.TTS_AUDIO_READY,
                source="tts",
                audio_data=result.audio_data,
                format=result.format,
            )
        except Exception as e:
            logger.warning(f"主 TTS 失败: {e}, 尝试离线备份")
            try:
                if self.tts_fallback and self.tts_fallback.is_available:
                    result = self.tts_fallback.synthesize(text)
                    self.event_bus.publish(
                        Event.TTS_AUDIO_READY,
                        source="tts_fallback",
                        audio_data=result.audio_data,
                        format=result.format,
                    )
                else:
                    raise RuntimeError("离线 TTS 不可用")
            except Exception as e2:
                logger.error(f"所有 TTS 均失败: {e2}")
                self.event_bus.publish(
                    Event.TTS_ERROR, source="tts", error=str(e2)
                )

    def _on_tts_audio_ready(self, event_data) -> None:
        """TTS 音频就绪: 播放"""
        audio_data = event_data.get("audio_data")
        audio_format = event_data.get("format", "mp3")

        if not audio_data:
            return

        # 切换到 SPEAKING 状态
        if not self.state_machine.transition(State.SPEAKING):
            logger.warning("无法切换到 SPEAKING 状态, 丢弃 TTS 输出")
            self.state_machine.force_idle()
            return

        # 异步播放 (独立线程, 不阻塞事件总线)
        self.event_bus.publish(Event.PLAYBACK_START, source="player")

        def _play_and_notify():
            try:
                self.audio_player.play(audio_data, format=audio_format)
                self.event_bus.publish(Event.PLAYBACK_DONE, source="player")
            except Exception as e:
                logger.error(f"播放失败: {e}")
                self.event_bus.publish(
                    Event.ERROR, source="player", message=f"播放失败: {e}"
                )

        threading.Thread(
            target=_play_and_notify, daemon=True, name="player-thread"
        ).start()

    def _on_tts_stream_chunk(self, event_data) -> None:
        """TTS 流式音频块: 队列播放"""
        audio_data = event_data.get("audio_data")
        if not audio_data:
            return

        audio_format = event_data.get("format", "mp3")
        sample_rate = event_data.get("sample_rate", 22050)

        if not self._stream_playback_started:
            if not self.state_machine.transition(State.SPEAKING):
                logger.warning("无法切换到 SPEAKING 状态, 丢弃流式 TTS")
                return
            self._stream_playback_started = True
            self.event_bus.publish(Event.PLAYBACK_START, source="player")
            fmt = "pcm" if audio_format == "pcm" else audio_format
            self.audio_player.start_stream(audio_format=fmt, sample_rate=sample_rate)

        self.audio_player.feed_stream_chunk(audio_data)

    def _on_playback_done(self, event_data) -> None:
        """播放完成: 回到 IDLE"""
        logger.info("播放完成, 回到 IDLE")
        self.state_machine.transition(State.IDLE)

    def _on_error(self, event_data) -> None:
        """通用错误处理"""
        source = event_data.source
        error_msg = event_data.get("error") or event_data.get("message", "未知错误")
        logger.error(f"[{source}] 错误: {error_msg}")

        # 对于严重错误, 强制回到 IDLE
        if self.state_machine.current_state != State.IDLE:
            # 尝试恢复
            if self.state_machine.current_state == State.SPEAKING:
                if self.audio_player:
                    self.audio_player.stop()
            self.state_machine.force_idle()

    # ================================================================
    # 键盘唤醒模式
    # ================================================================

    def enable_keyboard_wake_mode(self) -> None:
        """启用键盘唤醒模式: 按回车触发唤醒, 然后用语音对话"""
        self._keyboard_wake_mode = True
        self._stdin_thread = threading.Thread(
            target=self._keyboard_wake_loop,
            daemon=True,
            name="keyboard-wake",
        )
        self._stdin_thread.start()

    def _keyboard_wake_loop(self) -> None:
        """键盘唤醒读取循环"""
        while not self._running:
            time.sleep(0.1)

        print("=" * 50)
        print("⌨️  键盘唤醒模式已启用")
        print("   按 回车 唤醒我, 然后对麦克风说话")
        print("   输入 /quit 或 Ctrl+C 退出")
        print("=" * 50)

        try:
            while self._running:
                try:
                    user_input = input("\n⏎ 按回车唤醒 > ").strip()
                except (EOFError, KeyboardInterrupt):
                    break

                if user_input.lower() in ("/quit", "/exit", "/q"):
                    logger.info("用户请求退出")
                    self.stop()
                    break

                # 按回车触发唤醒 (不管输入什么都当作唤醒)
                if self.state_machine.current_state == State.SPEAKING and self.audio_player:
                    logger.info("打断当前播放 (barge-in)")
                    self.audio_player.stop()
                    self.event_bus.publish(Event.PLAYBACK_INTERRUPTED, source="engine")

                self._begin_utterance()

                if self.state_machine.transition(State.LISTENING):
                    logger.info("⌨️  键盘唤醒!")
                    self.event_bus.publish(
                        Event.WAKE_WORD_DETECTED,
                        source="keyboard_wake",
                        confidence=1.0,
                    )
                else:
                    logger.warning(f"无法唤醒, 当前状态: {self.state_machine.current_state.value}")
        except Exception as e:
            logger.error(f"键盘唤醒循环异常: {e}")

    # ================================================================
    # 键盘输入模式
    # ================================================================

    def enable_stdin_mode(self) -> None:
        """启用键盘输入模式: 在独立线程中读取 stdin, 直接处理文字指令"""
        self._stdin_mode = True
        self._stdin_thread = threading.Thread(
            target=self._stdin_loop,
            daemon=True,
            name="stdin-input",
        )
        self._stdin_thread.start()

    def _stdin_loop(self) -> None:
        """stdin 读取循环"""
        # 等待引擎启动
        while not self._running:
            time.sleep(0.1)

        print("=" * 50)
        print("📝 键盘输入模式已启用")
        print("   输入文字指令后按回车, 系统将直接处理")
        print("   输入 /quit 或 Ctrl+C 退出")
        print("=" * 50)

        try:
            while self._running:
                try:
                    user_input = input("\n💬 > ").strip()
                except (EOFError, KeyboardInterrupt):
                    break

                if not user_input:
                    continue

                if user_input.lower() in ("/quit", "/exit", "/q"):
                    logger.info("用户请求退出")
                    self.stop()
                    break

                # 直接处理文字输入 (模拟 ASR 结果)
                # 模拟唤醒流程: IDLE → LISTENING → THINKING
                logger.info(f"⌨️  文字输入: \"{user_input}\"")

                # 先切到 LISTENING (模拟唤醒)
                if self.state_machine.current_state == State.IDLE:
                    if not self.state_machine.transition(State.LISTENING):
                        logger.warning(f"无法切换到 LISTENING 状态")
                        continue

                # 再切到 THINKING (模拟语音结束)
                if self.state_machine.current_state == State.LISTENING:
                    if not self.state_machine.transition(State.THINKING):
                        logger.warning(f"无法切换到 THINKING 状态")
                        continue
                elif self.state_machine.current_state != State.THINKING:
                    logger.warning(
                        f"当前状态 {self.state_machine.current_state.value} "
                        f"无法处理输入")
                    continue

                self._handle_user_text(user_input)
        except Exception as e:
            logger.error(f"stdin 循环异常: {e}")

    # ================================================================
    # 主循环
    # ================================================================

    def run_forever(self) -> None:
        """启动并阻塞运行"""
        if not self._running:
            self.start()

        try:
            while self._running:
                # 检查状态超时
                self.state_machine.check_timeouts()
                # 主循环休眠 (事件驱动, 不需要高频轮询)
                time.sleep(0.1)
        except KeyboardInterrupt:
            logger.info("收到键盘中断")
        finally:
            self.stop()

    def start(self) -> None:
        """启动引擎"""
        if self._running:
            logger.warning("引擎已在运行中")
            return

        logger.info("启动 Smart Speaker 引擎...")
        self._running = True

        self._start_audio_worker()

        if self.audio_capture:
            saved = suppress_alsa_noise()
            try:
                self.audio_capture.start(self._audio_callback)
            finally:
                restore_alsa_noise(saved)
            logger.info("音频捕获已启动")

        logger.info("Smart Speaker 引擎启动完成 ✓")
        logger.info(f"唤醒词: \"{self.config.get('general.wake_word', '小智小智')}\"")
        logger.info("等待唤醒...")

    def stop(self) -> None:
        """优雅关闭"""
        if not self._running:
            return

        logger.info("正在关闭 Smart Speaker 引擎...")
        self._running = False

        self._stop_audio_worker()

        if self.audio_capture:
            self.audio_capture.stop()

        if self.audio_player:
            self.audio_player.stop()
            self.audio_player.release()

        if self.asr:
            self.asr.release()

        if self.wake_word_detector:
            self.wake_word_detector.release()

        if self.tts_piper:
            self.tts_piper.release()

        if self.web_server:
            self.web_server.stop()

        for event, handler in self._event_handlers:
            try:
                self.event_bus.unsubscribe(event, handler)
            except Exception:
                pass
        self._event_handlers.clear()

        logger.info("Smart Speaker 引擎已关闭")

    def _audio_callback(self, audio_frame: np.ndarray) -> None:
        """音频捕获回调 — 仅入队，重逻辑由 audio-worker 处理"""
        try:
            self._audio_queue.put_nowait(audio_frame)
        except queue.Full:
            try:
                self._audio_queue.get_nowait()
            except queue.Empty:
                pass
            try:
                self._audio_queue.put_nowait(audio_frame)
            except queue.Full:
                pass
        except Exception as e:
            logger.error(f"音频入队异常: {e}")

    def _signal_handler(self, signum, frame) -> None:
        """处理 SIGINT / SIGTERM"""
        logger.info(f"收到信号 {signal.Signals(signum).name}, 开始关闭...")
        self.stop()
        sys.exit(0)

    def _on_shutdown(self, event_data) -> None:
        """接收到 SHUTDOWN 事件"""
        self.stop()

    def _on_config_updated(self, event_data) -> None:
        """处理配置在线更新"""
        changed_keys = event_data.get("changed_keys", [])
        if not changed_keys:
            return
        logger.info(f"配置在线更新: {', '.join(changed_keys)}")

        restart_keys = []
        for key in changed_keys:
            if not self._apply_runtime_config(key):
                restart_keys.append(key)

        if restart_keys:
            logger.warning(
                f"以下配置需要重启才能生效: {', '.join(restart_keys)}"
            )

    def _apply_runtime_config(self, key: str) -> bool:
        """
        尝试在运行时应用一个配置变更。
        Returns: True 表示已热生效, False 表示需要重启。
        """
        value = self.config.get(key)

        # debug.save_audio — 直接赋值开关
        if key == "debug.save_audio":
            self._save_debug_audio_enabled = bool(value)
            return True

        # general.log_level — 更新全局日志级别
        if key == "general.log_level":
            import logging
            level = getattr(logging, str(value).upper(), logging.INFO)
            logging.getLogger("smart_speaker").setLevel(level)
            return True

        # VAD 参数 — 直接更新实例属性
        if key == "audio.vad.threshold" and self.vad:
            self.vad.threshold = float(value)
            return True
        if key == "audio.vad.min_speech_duration_ms" and self.vad:
            self.vad.min_speech_duration_ms = int(value)
            return True
        if key == "audio.vad.min_silence_duration_ms" and self.vad:
            self.vad.min_silence_duration_ms = int(value)
            return True

        # 这些 key 运行时每次都会从 self.config.get() 重读，
        # Phase 3b 已更新内存中的 Config._data，所以自动生效，无需额外操作。
        if key in (
            "wake_word.cooldown_ms",
            "audio.sample_rate",
            "audio.channels",
            "audio.vad.enabled",
            "general.name",
            "general.wake_word",
            "general.data_dir",
            "asr.preprocess",
            "asr.sherpa.num_threads",
            "tts.edge.voice",
            "tts.edge.rate",
            "tts.edge.pitch",
            "llm.max_tokens",
            "llm.temperature",
        ):
            return True

        # 其余所有配置需要重启
        return False

    def get_status(self) -> dict:
        """获取运行状态"""
        return {
            "state": self.state_machine.current_state.value,
            "uptime": "N/A",
            "capture_running": self.audio_capture.is_running if self.audio_capture else False,
            "player_playing": self.audio_player.is_playing if self.audio_player else False,
            "asr_available": self.asr.is_available if self.asr else False,
        }
