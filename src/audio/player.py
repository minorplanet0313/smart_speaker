"""
音频播放模块

支持播放 WAV/MP3 音频数据
- 阻塞播放 (play) 和异步播放 (play_async)
- 支持打断 (stop)
- 使用 PyAudio 和 soundfile 解码
"""

import io
import threading
import time
import wave
from typing import Optional

import numpy as np
import soundfile as sf

from src.utils.logger import get_logger

logger = get_logger(__name__)


class AudioPlayer:
    """
    音频播放器

    支持格式: WAV (PCM), MP3 (通过 soundfile)
    采样率自适应转换到 16kHz
    """

    def __init__(self, device_name: Optional[str] = None):
        self.device_name = device_name
        self._pa = None
        self._pyaudio = None
        self._is_playing = False
        self._stop_requested = False
        self._lock = threading.Lock()

    def _init_pyaudio(self) -> None:
        """延迟初始化"""
        if self._pa is not None:
            return
        import pyaudio
        self._pyaudio = pyaudio
        self._pa = pyaudio.PyAudio()

    def play(self, audio_data: bytes, format: str = "mp3") -> None:
        """
        阻塞播放音频

        Args:
            audio_data: 音频二进制数据
            format: 音频格式 ("mp3", "wav", "pcm")

        Raises:
            RuntimeError: 播放失败
        """
        with self._lock:
            self._is_playing = True
            self._stop_requested = False

        try:
            self._init_pyaudio()

            # 解码音频
            samples, sr = self._decode(audio_data, format)

            # 重采样到 16kHz (如果需要)
            if sr != 16000:
                samples = self._resample(samples, sr, 16000)
                sr = 16000

            # 确保是单声道
            if samples.ndim > 1:
                samples = samples.mean(axis=1)  # 立体声 → 单声道

            # 转换为 int16
            samples_int16 = (np.clip(samples, -1.0, 1.0) * 32767).astype(np.int16)

            # 打开输出流
            stream = self._pa.open(
                format=self._pyaudio.paInt16,
                channels=1,
                rate=sr,
                output=True,
                output_device_index=self._find_output_device(),
            )

            # 分块播放
            chunk_size = 4096
            for i in range(0, len(samples_int16), chunk_size):
                if self._stop_requested:
                    break
                chunk = samples_int16[i:i + chunk_size]
                stream.write(chunk.tobytes())

            stream.stop_stream()
            stream.close()

        except Exception as e:
            logger.error(f"播放失败: {e}")
            raise RuntimeError(f"音频播放失败: {e}")
        finally:
            with self._lock:
                self._is_playing = False
                self._stop_requested = False

    def play_async(self, audio_data: bytes, format: str = "mp3") -> threading.Thread:
        """
        异步播放

        Returns:
            播放线程
        """
        thread = threading.Thread(
            target=self.play,
            args=(audio_data, format),
            daemon=True,
            name="audio-player",
        )
        thread.start()
        return thread

    def stop(self) -> None:
        """打断当前播放"""
        if self._is_playing:
            logger.info("打断音频播放")
            self._stop_requested = True

    def wait_until_done(self, timeout: float = None) -> None:
        """等待播放完成"""
        deadline = time.time() + timeout if timeout else None
        while self._is_playing:
            if deadline and time.time() > deadline:
                break
            time.sleep(0.05)

    def _decode(self, audio_data: bytes, format: str) -> tuple:
        """
        解码音频数据

        Returns:
            (samples: np.ndarray, sample_rate: int)
        """
        if format in ("wav", "pcm"):
            # WAV/PCM 解码
            with wave.open(io.BytesIO(audio_data), 'rb') as wf:
                sr = wf.getframerate()
                frames = wf.readframes(wf.getnframes())
                samples = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
            return samples, sr
        else:
            # MP3 等格式 (通过 soundfile)
            with io.BytesIO(audio_data) as buf:
                samples, sr = sf.read(buf, dtype='float32')
            return samples, sr

    def _resample(self, samples: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
        """简单线性重采样"""
        if orig_sr == target_sr:
            return samples
        from scipy import signal
        duration = len(samples) / orig_sr
        num_target = int(duration * target_sr)
        return signal.resample(samples, num_target)

    def _find_output_device(self) -> Optional[int]:
        """查找输出设备"""
        if self.device_name:
            for i in range(self._pa.get_device_count()):
                info = self._pa.get_device_info_by_index(i)
                if (self.device_name in info.get("name", "") and
                        info.get("maxOutputChannels", 0) > 0):
                    return i
        return None

    @property
    def is_playing(self) -> bool:
        return self._is_playing

    @property
    def output_devices(self) -> list:
        """列出所有输出设备"""
        self._init_pyaudio()
        devices = []
        for i in range(self._pa.get_device_count()):
            info = self._pa.get_device_info_by_index(i)
            if info.get("maxOutputChannels", 0) > 0:
                devices.append({
                    "index": i,
                    "name": info["name"],
                    "channels": info["maxOutputChannels"],
                })
        return devices
