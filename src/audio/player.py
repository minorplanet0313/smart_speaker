"""
音频播放模块

支持播放 WAV/MP3 音频数据
- 阻塞播放 (play) 和异步播放 (play_async)
- 支持打断 (stop)
- 使用 PyAudio 和 soundfile 解码
"""

import queue
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
        self._stream_queue: Optional[queue.Queue] = None
        self._stream_thread: Optional[threading.Thread] = None
        self._stream_format = "wav"
        self._stream_sample_rate = 22050

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
            chunk_size = 2048  # 128ms @ 16kHz，避免 underrun
            stream = self._pa.open(
                format=self._pyaudio.paInt16,
                channels=1,
                rate=sr,
                output=True,
                frames_per_buffer=chunk_size,
                output_device_index=self._find_output_device(),
            )

            # 分块播放 (小块 = 低打断延迟)
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
        if self._stream_queue is not None:
            try:
                self._stream_queue.put_nowait(None)
            except queue.Full:
                pass

    def start_stream(self, audio_format: str = "wav", sample_rate: int = 22050) -> None:
        """开始流式播放会话"""
        with self._lock:
            self._stop_requested = False
            self._is_playing = True
            self._stream_format = audio_format
            self._stream_sample_rate = sample_rate
            self._stream_queue = queue.Queue(maxsize=32)
            self._stream_thread = threading.Thread(
                target=self._stream_play_loop,
                daemon=True,
                name="audio-stream-player",
            )
            self._stream_thread.start()

    def feed_stream_chunk(self, audio_data: bytes) -> None:
        """向流式播放队列送入音频块"""
        if self._stream_queue is not None and audio_data:
            self._stream_queue.put(audio_data)

    def end_stream(self) -> None:
        """结束流式播放会话"""
        if self._stream_queue is not None:
            self._stream_queue.put(None)

    def wait_stream_done(self, timeout: float = 60.0) -> None:
        """等待流式播放完成"""
        if self._stream_thread:
            self._stream_thread.join(timeout=timeout)

    def _stream_play_loop(self) -> None:
        """从队列读取音频块并连续播放"""
        try:
            self._init_pyaudio()
            stream = None
            pending = np.array([], dtype=np.float32)
            chunk_size = 2048  # 128ms @ 16kHz，避免 underrun

            while not self._stop_requested:
                try:
                    chunk = self._stream_queue.get(timeout=0.1)
                except queue.Empty:
                    continue
                if chunk is None:
                    break

                if self._stream_format == "pcm":
                    samples = (
                        np.frombuffer(chunk, dtype=np.int16).astype(np.float32) / 32768.0
                    )
                    sr = self._stream_sample_rate
                else:
                    samples, sr = self._decode(chunk, self._stream_format)

                if sr != 16000:
                    samples = self._resample(samples, sr, 16000)
                    sr = 16000
                if samples.ndim > 1:
                    samples = samples.mean(axis=1)
                if len(samples) == 0:
                    continue

                pending = np.concatenate([pending, samples]) if len(pending) else samples

                # 预缓冲至少 4 个 chunk (~512ms @16kHz) 再打开流，避免 underrun
                min_buffer = chunk_size * 4
                if stream is None and len(pending) >= min_buffer:
                    stream = self._pa.open(
                        format=self._pyaudio.paInt16,
                        channels=1,
                        rate=sr,
                        output=True,
                        frames_per_buffer=chunk_size,
                        output_device_index=self._find_output_device(),
                    )

                while stream and len(pending) >= chunk_size:
                    if self._stop_requested:
                        break
                    block = pending[:chunk_size]
                    pending = pending[chunk_size:]
                    int16 = (np.clip(block, -1.0, 1.0) * 32767).astype(np.int16)
                    stream.write(int16.tobytes())

                # 没数据可写时关闭流，避免持续空转 underrun
                if stream and len(pending) < chunk_size:
                    stream.stop_stream()
                    stream.close()
                    stream = None

            if stream and len(pending) > 0 and not self._stop_requested:
                int16 = (np.clip(pending, -1.0, 1.0) * 32767).astype(np.int16)
                stream.write(int16.tobytes())

            if stream:
                stream.stop_stream()
                stream.close()
        except Exception as e:
            logger.error(f"流式播放失败: {e}")
        finally:
            with self._lock:
                self._is_playing = False
                self._stop_requested = False
                self._stream_queue = None
                self._stream_thread = None

    def release(self) -> None:
        """释放 PyAudio 资源"""
        self.stop()
        if self._pa:
            try:
                self._pa.terminate()
            except Exception:
                pass
            self._pa = None
            self._pyaudio = None

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
        if orig_sr == target_sr or len(samples) == 0:
            return samples
        from scipy import signal
        duration = len(samples) / orig_sr
        num_target = max(1, int(duration * target_sr))
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
