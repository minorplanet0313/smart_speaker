"""
音频环形缓冲区 — 避免 np.append 的 O(n) 重分配
"""

from typing import Optional

import numpy as np


class AudioRingBuffer:
    """固定容量 float32 环形缓冲区，支持追加与按块消费"""

    def __init__(self, max_samples: int = 8192):
        self._data = np.zeros(max_samples, dtype=np.float32)
        self._max = max_samples
        self._count = 0

    def append(self, frame: np.ndarray) -> None:
        n = len(frame)
        if n == 0:
            return
        if n >= self._max:
            self._data[:] = frame[-self._max:]
            self._count = self._max
            return
        if self._count + n > self._max:
            shift = self._count + n - self._max
            self._data[: self._count - shift] = self._data[shift : self._count]
            self._count -= shift
        self._data[self._count : self._count + n] = frame
        self._count += n

    def consume(self, n: int) -> Optional[np.ndarray]:
        if self._count < n:
            return None
        chunk = self._data[:n].copy()
        if self._count > n:
            self._data[: self._count - n] = self._data[n : self._count]
        self._count -= n
        return chunk

    def __len__(self) -> int:
        return self._count

    def clear(self) -> None:
        self._count = 0
