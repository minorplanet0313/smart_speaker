"""
Piper TTS — 本地离线语音合成

MIT 开源协议, 完全本地运行
用于 Edge TTS 不可用时的离线备份

安装:
    pip install piper-tts

下载中文语音模型:
    wget https://huggingface.co/rhasspy/piper-voices/resolve/main/zh/zh_CN/huayan/medium/zh_CN-huayan-medium.onnx
    wget https://huggingface.co/rhasspy/piper-voices/resolve/main/zh/zh_CN/huayan/medium/zh_CN-huayan-medium.onnx.json
"""

import os
import time
from typing import Iterator, Optional

from src.tts.base import BaseTTS, TTSResult
from src.utils.logger import get_logger

logger = get_logger(__name__)


class PiperTTS(BaseTTS):
    """
    Piper 本地 TTS

    用于 Edge TTS 网络不可用时的降级方案
    """

    def __init__(
        self,
        model_path: str = "",
        config_path: Optional[str] = None,
    ):
        """
        Args:
            model_path: Piper 模型文件路径 (.onnx)
            config_path: 配置文件路径 (.onnx.json), 默认与模型同名
        """
        self.model_path = model_path
        self.config_path = config_path or (model_path + ".json" if model_path else "")
        self._available = self._check_model()
        self._voice_name = "piper-zh"
        self._voice = None  # 延迟加载缓存

    def _check_model(self) -> bool:
        """检查模型是否可用"""
        if not self.model_path or not os.path.exists(self.model_path):
            logger.info(f"Piper 模型未找到: {self.model_path}")
            return False

        if self.config_path and not os.path.exists(self.config_path):
            logger.warning(f"Piper 配置文件未找到: {self.config_path}")
            return False

        try:
            import piper.voice
            return True
        except ImportError:
            logger.info("piper-tts 未安装, 离线 TTS 不可用")
            return False

    def _load_voice(self):
        """延迟加载 Piper 语音模型 (只加载一次)"""
        if self._voice is not None:
            return self._voice
        import piper.voice
        logger.info(f"加载 Piper 模型: {self.model_path}")
        self._voice = piper.voice.PiperVoice.load(
            model_path=self.model_path,
            config_path=self.config_path,
        )
        return self._voice

    def synthesize(self, text: str) -> TTSResult:
        """
        合成语音

        Args:
            text: 要合成的文本

        Returns:
            TTSResult

        Raises:
            RuntimeError: 合成失败或模型不可用
        """
        if not self.is_available:
            raise RuntimeError("Piper TTS 不可用, 模型未加载")

        start_time = time.time()

        try:
            import io
            import wave

            # 加载语音模型 (首次调用时加载, 后续复用缓存)
            voice = self._load_voice()

            # 合成 (piper-tts >= 1.4: synthesize 返回 Iterable[AudioChunk])
            audio_data = bytearray()
            sample_rate = 22050  # 默认值
            for chunk in voice.synthesize(text):
                if hasattr(chunk, 'audio_int16_bytes'):
                    audio_data.extend(chunk.audio_int16_bytes)
                sample_rate = chunk.sample_rate

            # 写入 WAV 格式 (Piper 输出原始 PCM int16)
            wav_buf = io.BytesIO()
            with wave.open(wav_buf, 'wb') as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)  # 16-bit
                wf.setframerate(sample_rate)
                wf.writeframes(audio_data)

            latency_ms = (time.time() - start_time) * 1000

            logger.debug(f"Piper TTS 合成完成: \"{text[:50]}...\" ({latency_ms:.0f}ms)")

            return TTSResult(
                audio_data=wav_buf.getvalue(),
                format="wav",
                sample_rate=sample_rate,
                latency_ms=latency_ms,
            )

        except Exception as e:
            logger.error(f"Piper TTS 合成失败: {e}")
            raise RuntimeError(f"Piper TTS 合成失败: {e}")

    def synthesize_stream(self, text: str) -> Iterator[bytes]:
        """流式合成 (Piper 原生支持流式输出)"""
        if not self.is_available:
            return

        try:
            voice = self._load_voice()

            for chunk in voice.synthesize(text):
                if hasattr(chunk, 'audio_int16_bytes'):
                    yield chunk.audio_int16_bytes

        except Exception as e:
            logger.error(f"Piper 流式合成失败: {e}")

    @property
    def is_available(self) -> bool:
        return self._available

    @property
    def voice_name(self) -> str:
        return self._voice_name

    def release(self) -> None:
        """释放 Piper 模型缓存"""
        self._voice = None
