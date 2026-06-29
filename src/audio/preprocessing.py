"""
音频预处理管线

在 ASR 识别之前对原始语音音频进行标准化处理, 提升识别准确率。

处理步骤 (按顺序):
1. DC 偏移去除 — 消除直流偏置
2. 高通滤波 — 去除低频嗡声 (80Hz 以下)
3. 峰值归一化 — 统一音频响度到目标峰值

全部操作都在 float32 numpy 数组上完成, 零外部依赖 (仅需 scipy)。

使用示例:
    from src.audio.preprocessing import preprocess_pipeline
    audio = preprocess_pipeline(audio, sample_rate=16000)
"""

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np

from src.utils.logger import get_logger

logger = get_logger(__name__)

# 按 (sample_rate, cutoff_hz, order) 缓存 SOS 系数
_HIGHPASS_SOS_CACHE: Dict[Tuple[int, float, int], np.ndarray] = {}


def _get_highpass_sos(
    sample_rate: int,
    cutoff_hz: float,
    order: int,
) -> np.ndarray:
    key = (sample_rate, cutoff_hz, order)
    if key not in _HIGHPASS_SOS_CACHE:
        from scipy.signal import butter
        nyquist = sample_rate / 2.0
        _HIGHPASS_SOS_CACHE[key] = butter(
            order, cutoff_hz / nyquist, btype='high', output='sos'
        )
    return _HIGHPASS_SOS_CACHE[key]


@dataclass
class PreprocessConfig:
    """预处理配置"""
    enable_dc_removal: bool = True
    enable_highpass: bool = True
    highpass_cutoff_hz: float = 80.0
    enable_normalize: bool = True
    normalize_target_peak: float = 0.95
    enable_noise_reduce: bool = False  # 需要 noisereduce 库


def remove_dc_offset(signal: np.ndarray) -> np.ndarray:
    """
    去除 DC 偏移

    Args:
        signal: float32 数组

    Returns:
        去直流后的信号
    """
    return signal - np.mean(signal, dtype=np.float32)


def highpass_filter(
    signal: np.ndarray,
    sample_rate: int = 16000,
    cutoff_hz: float = 80.0,
    order: int = 4,
) -> np.ndarray:
    """
    Butterworth 高通滤波, 去除低频噪声 (嗡声/风噪)

    Args:
        signal: float32 数组
        sample_rate: 采样率
        cutoff_hz: 截止频率
        order: 滤波器阶数

    Returns:
        滤波后的信号
    """
    from scipy.signal import sosfilt

    nyquist = sample_rate / 2.0
    if cutoff_hz >= nyquist:
        logger.warning(f"高通截止频率 ({cutoff_hz}Hz) 超过奈奎斯特频率, 跳过")
        return signal

    sos = _get_highpass_sos(sample_rate, cutoff_hz, order)
    return sosfilt(sos, signal).astype(np.float32)


def peak_normalize(
    signal: np.ndarray,
    target_peak: float = 0.95,
) -> np.ndarray:
    """
    峰值归一化: 将音频幅度缩放到统一响度

    对静音帧不处理, 避免放大噪声。

    Args:
        signal: float32 数组
        target_peak: 目标峰值

    Returns:
        归一化后的信号
    """
    peak = np.max(np.abs(signal))
    if peak < 1e-6:
        # 静音段, 不放大噪声
        return signal
    gain = target_peak / peak
    return (signal * gain).astype(np.float32)


def noise_reduce(
    signal: np.ndarray,
    sample_rate: int = 16000,
) -> np.ndarray:
    """
    频谱噪声抑制 (需要 noisereduce 库)

    Args:
        signal: float32 数组
        sample_rate: 采样率

    Returns:
        降噪后的信号
    """
    try:
        import noisereduce as nr
        return nr.reduce_noise(
            y=signal,
            sr=sample_rate,
            stationary=True,
            prop_decrease=0.85,
        ).astype(np.float32)
    except ImportError:
        logger.warning("noisereduce 未安装, 跳过噪声抑制. pip install noisereduce")
        return signal
    except Exception as e:
        logger.error(f"噪声抑制失败: {e}")
        return signal


def preprocess_pipeline(
    signal: np.ndarray,
    sample_rate: int = 16000,
    config: Optional[PreprocessConfig] = None,
) -> np.ndarray:
    """
    完整的音频预处理管线

    Args:
        signal: float32 数组, range=[-1, 1], 单声道
        sample_rate: 采样率
        config: 预处理配置, 为 None 则使用默认配置

    Returns:
        处理后的信号 (与原信号长度相同)
    """
    if config is None:
        config = PreprocessConfig()

    # 输入校验
    if len(signal) == 0:
        return signal

    original_rms = float(np.sqrt(np.mean(np.square(signal))))

    # 1. DC 偏移去除
    if config.enable_dc_removal:
        dc_before = float(np.mean(signal))
        signal = remove_dc_offset(signal)
        if abs(dc_before) > 1e-4:
            logger.debug(f"DC 偏移去除: {dc_before:.6f} → {np.mean(signal):.6f}")

    # 2. 高通滤波
    if config.enable_highpass:
        signal = highpass_filter(signal, sample_rate, config.highpass_cutoff_hz)

    # 3. 噪声抑制 (可选, 需要 noisereduce)
    if config.enable_noise_reduce:
        signal = noise_reduce(signal, sample_rate)

    # 4. 峰值归一化
    if config.enable_normalize:
        signal = peak_normalize(signal, config.normalize_target_peak)

    processed_rms = float(np.sqrt(np.mean(np.square(signal))))
    logger.debug(
        f"音频预处理完成: RMS {original_rms:.4f} → {processed_rms:.4f}, "
        f"峰值 {np.max(np.abs(signal)):.3f}"
    )

    return signal
