"""
Vosk 本地 ASR 实现

基于 Vosk 的离线语音识别
- 模型: vosk-model-small-cn-0.22 (42MB)
- 内存占用: ~80-120MB
- 延迟: 150-400ms (Pi 3)
- 准确率: ~85% (中文短句)

安装:
    pip install vosk
    wget https://alphacephei.com/vosk/models/vosk-model-small-cn-0.22.zip
    unzip vosk-model-small-cn-0.22.zip -d models/
"""

import json
import os
import time
from typing import Optional

import numpy as np

from src.asr.base import ASRResult, BaseASR
from src.utils.logger import get_logger

logger = get_logger(__name__)


class VoskASR(BaseASR):
    """Vosk 离线语音识别"""

    def __init__(
        self,
        model_path: str = "models/vosk-model-small-cn-0.22",
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
        self._init_model()

    def _init_model(self) -> None:
        """初始化 Vosk 模型"""
        try:
            import vosk
            self._vosk = vosk

            if not os.path.exists(self.model_path):
                logger.warning(
                    f"Vosk 模型未找到: {self.model_path}\n"
                    f"请下载模型:\n"
                    f"  wget https://alphacephei.com/vosk/models/vosk-model-small-cn-0.22.zip\n"
                    f"  unzip vosk-model-small-cn-0.22.zip -d models/"
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
    ) -> ASRResult:
        """
        转写音频数据

        Args:
            audio_data: float32 数组, range=[-1, 1]
            sample_rate: 采样率

        Returns:
            ASRResult
        """
        start_time = time.time()

        if not self.is_available:
            raise RuntimeError("Vosk ASR 不可用")

        # 转换为 int16 PCM
        if audio_data.dtype == np.float32:
            audio_int16 = (np.clip(audio_data, -1.0, 1.0) * 32767).astype(np.int16)
        else:
            audio_int16 = audio_data.astype(np.int16)

        # 创建识别器
        recognizer = self._vosk.KaldiRecognizer(self._model, self.sample_rate)

        # 送入音频数据
        # Vosk 需要完整的音频数据, 分块送入
        chunk_size = 4096
        audio_bytes = audio_int16.tobytes()
        final_result = ""

        for i in range(0, len(audio_bytes), chunk_size * 2):
            chunk = audio_bytes[i:i + chunk_size * 2]
            if len(chunk) == 0:
                break
            if recognizer.AcceptWaveform(chunk):
                result = json.loads(recognizer.Result())
                final_result += result.get("text", "")
            # 也可以获取部分结果
            # partial = json.loads(recognizer.PartialResult())

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
        return self._model is not None

    def release(self) -> None:
        """释放 Vosk 模型资源"""
        if self._model:
            # Vosk Model 没有显式的 close 方法
            # Python GC 会自动处理
            self._model = None
            logger.info("Vosk 模型已释放")
