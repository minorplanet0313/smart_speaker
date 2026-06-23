"""
音频工具函数

- 格式转换 (int16 ↔ float32)
- 重采样
- 音频信息获取
"""

import numpy as np


def int16_to_float32(audio_int16: np.ndarray) -> np.ndarray:
    """
    将 int16 音频转换为 float32 [-1, 1]

    Args:
        audio_int16: dtype=int16 的音频数据

    Returns:
        dtype=float32, range=[-1, 1]
    """
    return audio_int16.astype(np.float32) / 32768.0


def float32_to_int16(audio_float32: np.ndarray) -> np.ndarray:
    """
    将 float32 [-1, 1] 音频转换为 int16

    Args:
        audio_float32: dtype=float32, range=[-1, 1]

    Returns:
        dtype=int16
    """
    # 防止削波
    audio_float32 = np.clip(audio_float32, -1.0, 1.0)
    return (audio_float32 * 32767).astype(np.int16)


def bytes_to_float32(audio_bytes: bytes) -> np.ndarray:
    """将原始字节转换为 float32 数组"""
    return int16_to_float32(
        np.frombuffer(audio_bytes, dtype=np.int16)
    )


def float32_to_bytes(audio_float32: np.ndarray) -> bytes:
    """将 float32 数组转换为原始字节"""
    return float32_to_int16(audio_float32).tobytes()


def compute_rms(audio: np.ndarray) -> float:
    """
    计算音频的 RMS (均方根) 能量

    用于简单的声音检测
    """
    if len(audio) == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(audio))))


def compute_db(audio: np.ndarray, ref: float = 1.0) -> float:
    """
    计算音频的分贝值

    Args:
        audio: 音频数据
        ref: 参考值 (float32 时为 1.0, int16 时为 32768)

    Returns:
        分贝值 (dB)
    """
    rms = compute_rms(audio)
    if rms < 1e-10:
        return -100.0
    return 20.0 * np.log10(rms / ref)
