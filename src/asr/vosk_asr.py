"""
Vosk 本地 ASR 实现

基于 Vosk 的离线语音识别
- 模型: vosk-model-cn-0.22 (1.3GB, 大模型, 准确率更高)
  - 小模型可选: vosk-model-small-cn-0.22 (42MB, ~85% 准确率, 更快)
- 内存占用: ~1.5-2GB (大模型) / ~80-120MB (小模型)
- 延迟: 300-800ms (大模型, Pi 3) / 150-400ms (小模型)
- 准确率: ~92% (中文短句, 大模型) / ~85% (小模型)

安装:
    pip install vosk
    # 大模型 (1.3GB)
    wget https://alphacephei.com/vosk/models/vosk-model-cn-0.22.zip
    unzip vosk-model-cn-0.22.zip -d models/
    # 小模型 (42MB, Pi 3B+ 推荐)
    wget https://alphacephei.com/vosk/models/vosk-model-small-cn-0.22.zip
    unzip vosk-model-small-cn-0.22.zip -d models/
"""

import json
import os
import time
from typing import Callable, Optional

import numpy as np

from src.asr.base import ASRResult, BaseASR
from src.utils.logger import get_logger

logger = get_logger(__name__)

# 部分结果回调类型: (partial_text: str) -> None
PartialCallback = Callable[[str], None]


class VoskASR(BaseASR):
    """Vosk 离线语音识别"""

    def __init__(
        self,
        model_path: str = "models/vosk-model-cn-0.22",
        sample_rate: int = 16000,
    ):
        """
        Args:
            model_path: Vosk 模型目录路径
            sample_rate: 采样率 (必须与模型匹配, 通常 16000)
        """
        self.model_path = model_path
        self.sample_rate = sample_rate
        self._model = None
        self._vosk = None
        self._recognizer = None
        self._model_loaded = False

    def _lazy_load_model(self) -> None:
        """延迟加载 Vosk 模型"""
        if self._model_loaded:
            return
        self._model_loaded = True
        self._init_model()

    def _init_model(self) -> None:
        """初始化 Vosk 模型"""
        try:
            import vosk
            vosk.SetLogLevel(-1)  # 静默 Kaldi 日志
            self._vosk = vosk

            if not os.path.exists(self.model_path):
                logger.warning(
                    f"Vosk 模型未找到: {self.model_path}\n"
                    f"请下载模型:\n"
                    f"  wget https://alphacephei.com/vosk/models/vosk-model-cn-0.22.zip\n"
                    f"  unzip vosk-model-cn-0.22.zip -d models/"
                )
                return

            logger.info(f"加载 Vosk 模型: {self.model_path} ...")
            self._model = vosk.Model(self.model_path)
            logger.info("Vosk 模型加载完成")

        except ImportError:
            logger.warning("vosk 未安装, 本地 ASR 不可用.\n"
                           "安装: pip install vosk")
            self._model = None
        except Exception as e:
            logger.error(f"Vosk 模型加载失败: {e}")
            self._model = None

    def transcribe(
        self,
        audio_data: np.ndarray,
        sample_rate: int = 16000,
        partial_callback: Optional[PartialCallback] = None,
    ) -> ASRResult:
        """
        转写音频数据

        Args:
            audio_data: float32 数组, range=[-1, 1]
            sample_rate: 采样率
            partial_callback: 可选, 流式部分结果回调.
                每处理一个音频块时调用 partial_callback(partial_text)

        Returns:
            ASRResult
        """
        start_time = time.time()

        if not self.is_available:
            raise RuntimeError("Vosk ASR 不可用")
        if audio_data.dtype == np.float32:
            audio_int16 = (np.clip(audio_data, -1.0, 1.0) * 32767).astype(np.int16)
        else:
            audio_int16 = audio_data.astype(np.int16)

        # 创建或复用识别器 (复用可保持上下文, 提升准确率)
        if self._recognizer is None:
            self._recognizer = self._vosk.KaldiRecognizer(self._model, self.sample_rate)
            self._recognizer.SetWords(True)
        else:
            self._recognizer.Reset()
            self._recognizer.SetWords(True)
        recognizer = self._recognizer

        # 送入音频数据
        # Vosk 需要完整的音频数据, 分块送入
        # 同时获取部分结果, 实现流式识别反馈
        chunk_size = 4096
        audio_bytes = audio_int16.tobytes()
        final_result = ""
        last_partial = ""

        for i in range(0, len(audio_bytes), chunk_size * 2):
            chunk = audio_bytes[i:i + chunk_size * 2]
            if len(chunk) == 0:
                break
            if recognizer.AcceptWaveform(chunk):
                result = json.loads(recognizer.Result())
                final_result += result.get("text", "")

            # 获取流式部分结果 (用户在说话过程中就能看到文字逐步出现)
            if partial_callback:
                partial = json.loads(recognizer.PartialResult())
                partial_text = partial.get("partial", "").strip()
                if partial_text and partial_text != last_partial:
                    last_partial = partial_text
                    partial_callback(partial_text)

        # 获取最终结果
        final = json.loads(recognizer.FinalResult())
        final_result += final.get("text", "")

        latency_ms = (time.time() - start_time) * 1000

        logger.debug(f"Vosk 识别完成: \"{final_result}\" ({latency_ms:.0f}ms)")

        return ASRResult(
            text=final_result.strip(),
            confidence=final.get("confidence", 0.0) if "confidence" in final else 0.85,
            is_final=True,
            latency_ms=latency_ms,
        )

    # ================================================================
    # 增量识别接口 (边听边识别, 降低感知延迟)
    # ================================================================

    def begin_utterance(self) -> None:
        """
        开始一个新的语音段识别.
        创建一个新的 KaldiRecognizer 实例用于增量识别.
        """
        if not self.is_available:
            raise RuntimeError("Vosk ASR 不可用")
        self._recognizer = self._vosk.KaldiRecognizer(self._model, self.sample_rate)
        self._recognizer.SetWords(True)

    def feed_chunk(
        self,
        audio_chunk: np.ndarray,
        partial_callback: Optional[PartialCallback] = None,
    ) -> Optional[str]:
        """
        向识别器送入一个音频块 (增量模式).

        在 LISTENING 状态下逐块送入音频, 不需要等待语音结束.
        与 begin_utterance() / end_utterance() 配合使用.

        Args:
            audio_chunk: float32 数组, range=[-1, 1]
            partial_callback: 可选部分结果回调

        Returns:
            如果 AcceptWaveform 接受了该块, 返回接受的文本; 否则 None
        """
        if self._recognizer is None:
            raise RuntimeError("请先调用 begin_utterance() 开始识别")

        # 转 int16
        if audio_chunk.dtype == np.float32:
            audio_int16 = (np.clip(audio_chunk, -1.0, 1.0) * 32767).astype(np.int16)
        else:
            audio_int16 = audio_chunk.astype(np.int16)

        audio_bytes = audio_int16.tobytes()
        accepted_text = None

        if self._recognizer.AcceptWaveform(audio_bytes):
            result = json.loads(self._recognizer.Result())
            accepted_text = result.get("text", "")

        # 部分结果
        if partial_callback:
            partial = json.loads(self._recognizer.PartialResult())
            partial_text = partial.get("partial", "").strip()
            if partial_text:
                partial_callback(partial_text)

        return accepted_text

    def end_utterance(self) -> str:
        """
        结束语音段识别, 获取最终结果.

        Returns:
            最终识别文本
        """
        if self._recognizer is None:
            raise RuntimeError("请先调用 begin_utterance() 开始识别")

        final = json.loads(self._recognizer.FinalResult())
        final_text = final.get("text", "")
        return final_text.strip()

    def transcribe_file(self, file_path: str) -> ASRResult:
        """转写音频文件"""
        import soundfile as sf
        audio, sr = sf.read(file_path, dtype='float32')
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        result = self.transcribe(audio, sr)
        return result

    @property
    def is_available(self) -> bool:
        self._lazy_load_model()
        return self._model is not None

    def release(self) -> None:
        """释放 Vosk 模型资源"""
        if self._model:
            # Vosk Model 没有显式的 close 方法
            # Python GC 会自动处理
            self._model = None
            logger.info("Vosk 模型已释放")
