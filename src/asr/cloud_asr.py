"""
云端 ASR 实现 (百度/阿里云)

当本地 Vosk ASR 不可用或识别质量不达标时使用
- 百度短语音识别: 免费 5万次/终身
- 阿里云一句话识别: 3个月试用

Usage:
    asr = CloudASR(provider="baidu", api_key="...", secret_key="...")
    result = asr.transcribe(audio_data)
"""

import base64
import io
import json
import time
import wave
from typing import Optional
from urllib.request import Request, urlopen

import numpy as np

from src.asr.base import ASRResult, BaseASR
from src.utils.logger import get_logger

logger = get_logger(__name__)


class CloudASR(BaseASR):
    """
    云端 ASR

    支持:
    - 百度 AI 短语音识别 (REST)
    - 阿里云 NLS 一句话识别 (WebSocket, 待完善)
    """

    # 百度 ASR API 地址
    BAIDU_TOKEN_URL = "https://aip.baidubce.com/oauth/2.0/token"
    BAIDU_ASR_URL = "https://vop.baidu.com/server_api"

    # 阿里云 ASR
    ALIYUN_ASR_URL = "https://nls-gateway.cn-shanghai.aliyuncs.com/stream/v1/asr"

    def __init__(
        self,
        provider: str = "baidu",
        api_key: str = "",
        secret_key: str = "",
    ):
        """
        Args:
            provider: "baidu" 或 "aliyun"
            api_key: 百度 API Key / 阿里云 AccessKey
            secret_key: 百度 Secret Key / 阿里云 AccessKey Secret
        """
        self.provider = provider
        self.api_key = api_key
        self.secret_key = secret_key
        self._token: Optional[str] = None
        self._token_expiry: float = 0.0

        if provider == "baidu":
            self._available = bool(api_key and secret_key)
        elif provider == "aliyun":
            self._available = bool(api_key and secret_key)
        else:
            self._available = False

        if self._available:
            logger.info(f"云端 ASR ({provider}) 初始化完成")
        else:
            logger.info(f"云端 ASR ({provider}) 未配置 API Key, 跳过")

    def transcribe(
        self,
        audio_data: np.ndarray,
        sample_rate: int = 16000,
    ) -> ASRResult:
        start_time = time.time()

        if not self.is_available:
            raise RuntimeError(f"云端 ASR ({self.provider}) 未配置")

        if self.provider == "baidu":
            result = self._transcribe_baidu(audio_data, sample_rate)
        elif self.provider == "aliyun":
            result = self._transcribe_aliyun(audio_data, sample_rate)
        else:
            raise RuntimeError(f"不支持的 ASR 提供商: {self.provider}")

        latency_ms = (time.time() - start_time) * 1000

        logger.info(
            f"云端 ASR ({self.provider}) 完成: \"{result.get('text', '')}\" "
            f"({latency_ms:.0f}ms)"
        )

        return ASRResult(
            text=result.get("text", ""),
            confidence=result.get("confidence", 0.9),
            latency_ms=latency_ms,
        )

    def _transcribe_baidu(
        self,
        audio_data: np.ndarray,
        sample_rate: int,
        max_retries: int = 2,
    ) -> dict:
        """百度 ASR 识别 (含重试)"""
        # 转换为 int16 PCM
        if audio_data.dtype == np.float32:
            audio_int16 = (np.clip(audio_data, -1.0, 1.0) * 32767).astype(np.int16)
        else:
            audio_int16 = audio_data.astype(np.int16)

        # 编码为 PCM WAV base64
        wav_buf = io.BytesIO()
        with wave.open(wav_buf, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(audio_int16.tobytes())

        audio_b64 = base64.b64encode(wav_buf.getvalue()).decode()

        last_error = None
        for attempt in range(max_retries + 1):
            try:
                # 获取 access token (可能因过期而刷新)
                token = self._get_baidu_token()

                # 构造请求
                payload = json.dumps({
                    "format": "pcm",
                    "rate": sample_rate,
                    "channel": 1,
                    "cuid": "smart_speaker",
                    "token": token,
                    "speech": audio_b64,
                    "len": len(audio_int16) * 2,
                }).encode('utf-8')

                req = Request(
                    self.BAIDU_ASR_URL,
                    data=payload,
                    headers={"Content-Type": "application/json"},
                )

                with urlopen(req, timeout=10) as resp:
                    data = json.loads(resp.read().decode())

                err_no = data.get("err_no", -1)
                if err_no == 0:
                    text = " ".join(data.get("result", []))
                    return {"text": text, "confidence": 0.9}
                else:
                    # Token 过期/无效时刷新重试
                    if err_no in (3301, 3302, 3303, 3310, 3311) and attempt < max_retries:
                        logger.warning(f"百度 ASR Token 问题 (err_no={err_no}), "
                                       f"刷新重试 ({attempt + 1}/{max_retries})...")
                        self._token = None  # 强制刷新 token
                        self._token_expiry = 0.0
                        continue
                    raise RuntimeError(
                        f"百度 ASR 错误: err_no={err_no}, "
                        f"msg={data.get('err_msg', 'unknown')}"
                    )

            except (IOError, OSError, TimeoutError) as e:
                last_error = e
                if attempt < max_retries:
                    backoff = (attempt + 1) * 0.5
                    logger.warning(f"百度 ASR 网络错误: {e}, "
                                   f"重试 ({attempt + 1}/{max_retries}, "
                                   f"等待 {backoff}s)...")
                    time.sleep(backoff)
                else:
                    raise RuntimeError(f"百度 ASR 网络请求失败: {e}") from e

            except RuntimeError:
                raise

        # 不应到达这里
        raise RuntimeError(f"百度 ASR 失败: {last_error}")

    def _transcribe_aliyun(
        self,
        audio_data: np.ndarray,
        sample_rate: int,
    ) -> dict:
        """阿里云 ASR 识别 (简化版)"""
        # 阿里云一句话识别需要 WebSocket
        # 这里提供一个简化版 HTTP 实现
        # 完整实现建议使用阿里云 SDK: pip install alibabacloud-nls

        # 转换为 int16
        if audio_data.dtype == np.float32:
            audio_int16 = (np.clip(audio_data, -1.0, 1.0) * 32767).astype(np.int16)
        else:
            audio_int16 = audio_data.astype(np.int16)

        # TODO: 实现完整的阿里云 NLS 一句话识别
        # 参考: https://help.aliyun.com/document_detail/84424.html
        raise NotImplementedError(
            "阿里云 ASR 完整实现请使用: pip install alibabacloud-nls"
        )

    def _get_baidu_token(self) -> str:
        """获取百度 API access token"""
        if self._token and time.time() < self._token_expiry:
            return self._token

        url = (f"{self.BAIDU_TOKEN_URL}?"
               f"grant_type=client_credentials"
               f"&client_id={self.api_key}"
               f"&client_secret={self.secret_key}")

        req = Request(url)
        with urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read().decode())

        self._token = data.get("access_token")
        # Token 有效期 30 天, 但我们提前刷新
        self._token_expiry = time.time() + data.get("expires_in", 2592000) - 3600

        if not self._token:
            raise RuntimeError(f"百度 Token 获取失败: {data}")

        logger.debug("百度 ASR Token 已获取")
        return self._token

    def transcribe_file(self, file_path: str) -> ASRResult:
        import soundfile as sf
        audio, sr = sf.read(file_path, dtype='float32')
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        return self.transcribe(audio, sr)

    @property
    def is_available(self) -> bool:
        return self._available

    def release(self) -> None:
        pass
