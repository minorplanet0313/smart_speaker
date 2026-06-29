"""
主引擎 — Smart Speaker 的核心协调器

职责:
1. 创建和初始化所有模块
2. 连接事件总线, 协调模块间数据流
3. 管理主循环和生命周期
4. 异常处理和优雅关闭
"""

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
from src.asr.base import ASRResult
from src.asr.vosk_asr import VoskASR
from src.asr.cloud_asr import CloudASR
from src.audio.preprocessing import PreprocessConfig, preprocess_pipeline
from src.llm.deepseek import DeepSeekLLM
from src.llm.context import ConversationContext
from src.tts.edge_tts import EdgeTTS
from src.tts.piper_tts import PiperTTS
from src.skills.base import SkillContext, SkillResult
from src.skills.skill_manager import SkillManager
from src.skills.builtin.chat_skill import ChatSkill
from src.skills.builtin.time_skill import TimeSkill
from src.utils.config import Config, get_config
from src.utils.logger import get_logger, setup_logger

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
        self.conversation_context: Optional[ConversationContext] = None
        self.skill_manager: Optional[SkillManager] = None

        # 运行时状态
        self._running = False
        self._audio_buffer = bytearray()
        self._speech_audio_buffer: list = []  # 当前语音片段的音频块
        self._wake_word_cooldown_until = 0.0
        self._asr_incremental_mode = True  # 边听边识别模式 (降低感知延迟)
        self._asr_preprocess = True        # 音频预处理 (提升识别准确率)
        self._asr_final_text = ""  # 增量 ASR 累积的最终文本

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
        """初始化唤醒词检测"""
        ww_config = self.config.get("wake_word", {})
        if ww_config.get("enabled", True):
            self.wake_word_detector = WakeWordDetector(
                model_path=ww_config.get("model_path", ""),
                threshold=ww_config.get("threshold", 0.5),
                inference_framework=ww_config.get("inference_framework", "onnx"),
            )
            logger.info(f"唤醒词检测 (openWakeWord) 初始化完成, "
                        f"threshold={ww_config.get('threshold', 0.5)}")

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
        self.llm = DeepSeekLLM(
            api_key=llm_config.get("api_key", ""),
            model=llm_config.get("model", "deepseek-chat"),
            base_url=llm_config.get("base_url", "https://api.deepseek.com"),
            system_prompt=self.config.get("conversation.system_prompt", ""),
            temperature=llm_config.get("temperature", 0.7),
            max_tokens=llm_config.get("max_tokens", 1024),
            timeout=llm_config.get("timeout_seconds", 30),
        )
        logger.info(f"LLM (DeepSeek) 初始化完成: model={llm_config.get('model')}")

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

    def _init_skills(self) -> None:
        """初始化技能管理器"""
        self.skill_manager = SkillManager()
        # 注册内置技能
        self.skill_manager.register(ChatSkill(self.llm))
        self.skill_manager.register(TimeSkill())
        # 天气技能需要 API key
        weather_api_key = self.config.get("skills.weather.api_key", "")
        if weather_api_key:
            from src.skills.builtin.weather_skill import WeatherSkill
            self.skill_manager.register(WeatherSkill(weather_api_key))
        logger.info(f"技能管理器初始化完成: "
                     f"{len(self.skill_manager.list_skills())} 个技能已注册")

    def _init_conversation_context(self) -> None:
        """初始化对话上下文管理"""
        conv_config = self.config.get("conversation", {})
        self.conversation_context = ConversationContext(
            max_history_rounds=conv_config.get("max_history_rounds", 20),
            context_timeout_seconds=conv_config.get("context_timeout_seconds", 300),
        )
        logger.info(f"对话上下文管理初始化完成: "
                     f"max_rounds={conv_config.get('max_history_rounds', 20)}")

    # ================================================================
    # 事件连接
    # ================================================================

    def _wire_events(self) -> None:
        """连接模块间的事件"""
        logger.info("连接事件总线...")

        # 音频帧 → 唤醒词检测 + VAD
        self.event_bus.subscribe(Event.AUDIO_FRAME, self._on_audio_frame)

        # 唤醒词 → 开始监听
        if self.wake_word_detector:
            self.wake_word_detector.on_detected(self._on_wake_word_detected)

        # 语音结束 → ASR
        self.event_bus.subscribe(Event.SPEECH_END, self._on_speech_end)

        # ASR 结果 → 技能处理 → LLM
        self.event_bus.subscribe(Event.ASR_RESULT, self._on_asr_result)
        self.event_bus.subscribe(Event.ASR_PARTIAL, self._on_asr_partial)

        # LLM 流式输出
        self.event_bus.subscribe(Event.LLM_STREAM_CHUNK, self._on_llm_stream_chunk)

        # LLM 完成 → TTS
        self.event_bus.subscribe(Event.LLM_RESPONSE, self._on_llm_response)

        # TTS 就绪 → 播放
        self.event_bus.subscribe(Event.TTS_AUDIO_READY, self._on_tts_audio_ready)

        # 播放完成 → 回到 IDLE
        self.event_bus.subscribe(Event.PLAYBACK_DONE, self._on_playback_done)

        # 各类错误
        self.event_bus.subscribe(Event.ASR_ERROR, self._on_error)
        self.event_bus.subscribe(Event.LLM_ERROR, self._on_error)
        self.event_bus.subscribe(Event.TTS_ERROR, self._on_error)
        self.event_bus.subscribe(Event.AUDIO_DEVICE_ERROR, self._on_error)

        # 关闭信号
        self.event_bus.subscribe(Event.SHUTDOWN, self._on_shutdown)

        logger.info("事件连接完成")

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

    def _on_audio_frame(self, event_data) -> None:
        """处理音频帧: 路由到唤醒词检测器 (IDLE时) 或 VAD (LISTENING时)"""
        audio_frame = event_data.get("frame")
        if audio_frame is None:
            return

        current_state = self.state_machine.current_state

        if current_state == State.IDLE and self.wake_word_detector:
            # 检查冷却时间
            now = time.time()
            if now < self._wake_word_cooldown_until:
                return
            # 每 ~5 秒打印音频电平 (确认麦克风在工作)
            if not hasattr(self, '_audio_level_debug_time'):
                self._audio_level_debug_time = 0.0
                self._audio_level_samples = 0
                self._audio_level_rms_sum = 0.0
            self._audio_level_samples += 1
            self._audio_level_rms_sum += float(
                np.sqrt(np.mean(np.square(audio_frame)))
            )
            if now - self._audio_level_debug_time >= 5.0:
                avg_rms = self._audio_level_rms_sum / max(self._audio_level_samples, 1)
                logger.info(
                    f"🎤 音频电平: RMS={avg_rms:.4f} "
                    f"({'正常' if avg_rms > 0.001 else '⚠️ 太弱/无声!'}) "
                    f"[{self._audio_level_samples} 帧/5s]"
                )
                self._audio_level_debug_time = now
                self._audio_level_samples = 0
                self._audio_level_rms_sum = 0.0
            # 喂给唤醒词检测器
            self.wake_word_detector.detect(audio_frame)

        elif current_state == State.LISTENING:
            # 累积音频缓冲 (用于保存调试录音和后备 ASR)
            self._speech_audio_buffer.append(audio_frame)
            # 增量 ASR: 边听边识别, 逐帧送入 Vosk
            if self._asr_incremental_mode and self.asr and self.asr.is_available:
                try:
                    accepted = self.asr.feed_chunk(audio_frame)
                    if accepted:
                        self._asr_final_text += accepted
                except Exception:
                    pass  # 增量送入失败不影响主流程
            # VAD 处理
            if self.vad:
                vad_state = self.vad.process(audio_frame)
                if vad_state == VADState.SPEECH_START:
                    self.event_bus.publish(Event.SPEECH_START, source="vad")
                elif vad_state == VADState.SPEECH_END:
                    self.event_bus.publish(Event.SPEECH_END, source="vad")

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

        # 清空之前的语音缓冲
        self._speech_audio_buffer = []

        # 增量 ASR 模式: 提前创建识别器, 边听边识别
        self._asr_final_text = ""
        if self._asr_incremental_mode and self.asr and self.asr.is_available:
            try:
                self.asr.begin_utterance()
            except Exception as e:
                logger.warning(f"增量 ASR 初始化失败: {e}")

        # 切换到 LISTENING
        if self.state_machine.transition(State.LISTENING):
            self.event_bus.publish(
                Event.WAKE_WORD_DETECTED,
                source="wake_word",
                confidence=confidence,
            )

    def _on_speech_end(self, event_data) -> None:
        """语音结束: 切换到 THINKING, 触发 ASR (含音频预处理)"""
        if not self.state_machine.is_in(State.LISTENING):
            return

        logger.info(f"语音结束, 累积了 {len(self._speech_audio_buffer)} 个音频块")

        # 检查是否有足够的语音
        if not self._speech_audio_buffer:
            logger.info("语音缓冲为空, 回到 IDLE")
            self.state_machine.transition(State.IDLE)
            return

        self.state_machine.transition(State.THINKING)

        # 合并音频数据
        audio_data = np.concatenate(self._speech_audio_buffer)
        self._speech_audio_buffer = []

        # 音频预处理: 去直流 + 高通滤波 + 峰值归一化 (提升 ASR 准确率)
        if self._asr_preprocess:
            audio_data = preprocess_pipeline(audio_data, sample_rate=16000)

        # 保存录制音频到文件 (调试用)
        self._save_debug_audio(audio_data)

        # 选择 ASR 模式: 增量模式 (边听边识别) 或 批量模式
        if self._asr_incremental_mode and self.asr and self.asr.is_available:
            # 增量模式: 音频已逐帧送入, 直接获取最终结果
            threading.Thread(
                target=self._do_asr_finalize,
                args=(audio_data,),
                daemon=True,
                name="asr-thread",
            ).start()
        else:
            # 批量模式: 一次性送入全部音频
            threading.Thread(
                target=self._do_asr,
                args=(audio_data,),
                daemon=True,
                name="asr-thread",
            ).start()

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

    def _on_asr_result(self, event_data) -> None:
        """ASR 结果: 通过技能管理器处理"""
        text = event_data.get("text", "").strip()

        if not text:
            # 空结果, 回到 IDLE (播放提示音)
            logger.info("ASR 返回空文本, 回到 IDLE")
            self.state_machine.transition(State.IDLE)
            return

        # 通过技能管理器找到合适的技能
        context = SkillContext(
            conversation_id=self.conversation_context.current_conversation_id,
        )
        result = self.skill_manager.execute(text, context)

        # 技能返回的响应文本
        if result and result.success and result.response_text:
            # 技能直接返回文本, 不需要 LLM
            response_text = result.response_text
            logger.info(f"技能直接响应: \"{response_text}\"")
            self._do_tts(response_text)
        elif result and not result.success:
            # 技能失败, 使用错误提示
            logger.error(f"技能执行失败: {result.error_message}")
            self._do_tts("抱歉, 出了点问题, 请再试一次")
        elif result and result.success and result.needs_llm:
            # ChatSkill 兜底: 需要 LLM 处理
            self._do_llm(text)
        else:
            # 无法处理
            logger.warning(f"未找到合适的技能处理: \"{text}\"")
            self._do_tts("抱歉, 我不太明白你的意思")

    def _do_asr_finalize(self, audio_data: np.ndarray) -> None:
        """
        增量模式 ASR 最终化 (在独立线程中).
        音频已在 LISTENING 期间逐帧送入, 这里只需获取最终结果.
        失败时依次降级: 批量 Vosk → 云端 ASR.
        """
        start_time = time.time()
        try:
            final_text = self.asr.end_utterance()
            # 合并增量模式中 AcceptWaveform 累积的文本
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
        """LLM 完成响应: 触发 TTS"""
        text = event_data.get("text", "").strip()
        if not text:
            logger.warning("LLM 返回空文本")
            self.state_machine.transition(State.IDLE)
            return

        logger.info(f"LLM 响应: \"{text[:100]}{'...' if len(text) > 100 else ''}\"")
        self._do_tts(text)

    def _save_debug_audio(self, audio_data: np.ndarray) -> None:
        """保存录制音频到文件 (调试用)"""
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
            # 添加用户消息到上下文
            self.conversation_context.add_user_message(text)

            # 获取历史消息
            messages = self.conversation_context.get_messages()

            # 调用 LLM (流式)
            logger.info(f"调用 LLM: \"{text[:50]}...\"")
            full_response = ""
            for delta in self.llm.chat_stream(messages):
                full_response += delta

            if not full_response:
                logger.warning("LLM 返回空响应")
                self.state_machine.transition(State.IDLE)
                return

            logger.info(f"LLM 响应: \"{full_response[:100]}...\"")

            # 添加助手消息到上下文
            self.conversation_context.add_assistant_message(full_response)

            # 发布 LLM 响应事件
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
            # 尝试主 TTS (Edge)
            result = self.tts_edge.synthesize(text)
            logger.info(f"TTS 完成: {result.latency_ms:.0f}ms, "
                         f"{len(result.audio_data)} bytes")
            self.event_bus.publish(
                Event.TTS_AUDIO_READY,
                source="tts",
                audio_data=result.audio_data,
                format=result.format,
            )
        except Exception as e:
            logger.warning(f"主 TTS (Edge) 失败: {e}, 尝试离线备份")
            try:
                if self.tts_piper.is_available:
                    result = self.tts_piper.synthesize(text)
                    self.event_bus.publish(
                        Event.TTS_AUDIO_READY,
                        source="tts_piper",
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
            return

        # 播放音频
        self.event_bus.publish(Event.PLAYBACK_START, source="player")
        try:
            self.audio_player.play(audio_data, format=audio_format)
            self.event_bus.publish(Event.PLAYBACK_DONE, source="player")
        except Exception as e:
            logger.error(f"播放失败: {e}")
            self.event_bus.publish(
                Event.ERROR, source="player", message=f"播放失败: {e}"
            )

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

                self._speech_audio_buffer = []

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

                # 通过技能管理器处理
                context = SkillContext(
                    conversation_id=self.conversation_context.current_conversation_id,
                )
                result = self.skill_manager.execute(user_input, context)

                if result and result.success and result.response_text:
                    # 技能直接返回响应, 不需要 LLM
                    logger.info(f"技能直接响应: \"{result.response_text}\"")
                    self._do_tts(result.response_text)
                elif result and not result.success:
                    logger.error(f"技能执行失败: {result.error_message}")
                    self._do_tts("抱歉, 出了点问题, 请再试一次")
                elif result and result.success and result.needs_llm:
                    # ChatSkill 兜底: 需要 LLM 处理
                    self._do_llm(user_input)
                else:
                    logger.warning(f"无法处理输入: \"{user_input}\"")
                    self._do_tts("抱歉, 我不太明白你的意思")
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

        # 启动音频捕获
        if self.audio_capture:
            self.audio_capture.start(self._audio_callback)
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

        # 停止音频捕获
        if self.audio_capture:
            self.audio_capture.stop()

        # 停止播放
        if self.audio_player:
            self.audio_player.stop()

        # 释放资源
        if self.asr:
            self.asr.release()

        logger.info("Smart Speaker 引擎已关闭")

    def _audio_callback(self, audio_frame: np.ndarray) -> None:
        """
        音频捕获回调 — 在音频线程中调用
        将音频帧发布到事件总线
        """
        try:
            self.event_bus.publish(
                Event.AUDIO_FRAME,
                source="audio_capture",
                frame=audio_frame,
            )
        except Exception as e:
            logger.error(f"音频回调异常: {e}")

    def _signal_handler(self, signum, frame) -> None:
        """处理 SIGINT / SIGTERM"""
        logger.info(f"收到信号 {signal.Signals(signum).name}, 开始关闭...")
        self.stop()
        sys.exit(0)

    def _on_shutdown(self, event_data) -> None:
        """接收到 SHUTDOWN 事件"""
        self.stop()

    def get_status(self) -> dict:
        """获取运行状态"""
        return {
            "state": self.state_machine.current_state.value,
            "uptime": "N/A",
            "capture_running": self.audio_capture.is_running if self.audio_capture else False,
            "player_playing": self.audio_player.is_playing if self.audio_player else False,
            "asr_available": self.asr.is_available if self.asr else False,
        }
