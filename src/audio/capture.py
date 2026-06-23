"""
音频捕获模块

基于 PyAudio 的流式音频采集
- 16kHz, 16bit, 单声道
- 回调模式: 音频帧通过 callback 传递给消费者
- 自动检测和恢复 USB 设备
"""

import threading
import time
from typing import Callable, Optional

import numpy as np

from src.audio.utils import int16_to_float32
from src.utils.logger import get_logger

logger = get_logger(__name__)


class AudioCapture:
    """
    流式音频捕获

    Usage:
        capture = AudioCapture(sample_rate=16000)
        capture.start(callback=my_handler)
        ...
        capture.stop()
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        channels: int = 1,
        chunk_size: int = 1024,
        device_name: Optional[str] = None,
    ):
        self.sample_rate = sample_rate
        self.channels = channels
        self.chunk_size = chunk_size
        self.device_name = device_name

        self._pyaudio = None
        self._stream = None
        self._callback: Optional[Callable[[np.ndarray], None]] = None
        self._is_running = False
        self._lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None

    def start(self, callback: Callable[[np.ndarray], None]) -> None:
        """
        启动音频捕获

        Args:
            callback: 回调函数, 接收 np.ndarray (float32, shape=(chunk_size,))
        """
        with self._lock:
            if self._is_running:
                logger.warning("音频捕获已在运行")
                return

            self._callback = callback
            self._init_pyaudio()
            self._is_running = True
            self._thread = threading.Thread(
                target=self._read_loop,
                daemon=True,
                name="audio-capture",
            )
            self._thread.start()
            logger.info(f"音频捕获已启动: {self.sample_rate}Hz, "
                         f"{self.channels}ch, chunk={self.chunk_size}")

    def _init_pyaudio(self) -> None:
        """延迟初始化 PyAudio (避免导入时的副作用)"""
        import pyaudio
        self._pyaudio = pyaudio
        self._pa = pyaudio.PyAudio()

        # 查找输入设备
        device_index = None
        if self.device_name:
            for i in range(self._pa.get_device_count()):
                info = self._pa.get_device_info_by_index(i)
                if (self.device_name in info.get("name", "") and
                        info.get("maxInputChannels", 0) > 0):
                    device_index = i
                    logger.info(f"使用指定麦克风: {info['name']} (index={i})")
                    break
            if device_index is None:
                logger.warning(f"未找到麦克风设备 '{self.device_name}', "
                               f"使用系统默认")

        # 打开流
        self._stream = self._pa.open(
            format=pyaudio.paInt16,
            channels=self.channels,
            rate=self.sample_rate,
            input=True,
            input_device_index=device_index,
            frames_per_buffer=self.chunk_size,
            stream_callback=None,  # 使用阻塞模式, 在独立线程中读取
        )

    def _read_loop(self) -> None:
        """音频读取循环 (运行在独立线程)"""
        while self._is_running:
            try:
                # 读取音频块
                data = self._stream.read(
                    self.chunk_size,
                    exception_on_overflow=False,
                )
                # 转换为 float32
                audio_frame = int16_to_float32(
                    np.frombuffer(data, dtype=np.int16)
                )
                # 调用回调
                if self._callback:
                    self._callback(audio_frame)

            except OSError as e:
                logger.error(f"音频读取错误: {e}")
                if self._is_running:
                    logger.info("尝试重新连接音频设备...")
                    time.sleep(1)
                    try:
                        self._reconnect()
                    except Exception as re:
                        logger.error(f"音频设备重连失败: {re}")
                        time.sleep(3)
            except Exception as e:
                logger.error(f"音频捕获异常: {e}", exc_info=True)
                if self._is_running:
                    time.sleep(0.5)

    def _reconnect(self) -> None:
        """重新连接音频设备 (USB 热插拔恢复)"""
        if self._stream:
            try:
                self._stream.stop_stream()
                self._stream.close()
            except Exception:
                pass
        self._init_pyaudio()
        logger.info("音频设备已重新连接")

    def stop(self) -> None:
        """停止音频捕获"""
        with self._lock:
            if not self._is_running:
                return
            self._is_running = False

        if self._thread:
            self._thread.join(timeout=2)

        if self._stream:
            try:
                self._stream.stop_stream()
                self._stream.close()
            except Exception:
                pass

        if self._pa:
            try:
                self._pa.terminate()
            except Exception:
                pass

        logger.info("音频捕获已停止")

    @property
    def is_running(self) -> bool:
        return self._is_running

    @property
    def input_devices(self) -> list:
        """列出所有输入设备"""
        import pyaudio
        pa = pyaudio.PyAudio()
        devices = []
        for i in range(pa.get_device_count()):
            info = pa.get_device_info_by_index(i)
            if info.get("maxInputChannels", 0) > 0:
                devices.append({
                    "index": i,
                    "name": info["name"],
                    "channels": info["maxInputChannels"],
                    "sample_rate": int(info.get("defaultSampleRate", 0)),
                })
        pa.terminate()
        return devices
