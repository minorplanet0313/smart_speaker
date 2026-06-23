"""
Microsoft Edge TTS 实现

基于 edge-tts 库, 调用微软 Edge 浏览器免费 TTS 服务
- 完全免费, 无需注册
- 中文语音质量优秀 (30+ 种神经语音)
- 支持 SSML
- 需要联网

安装:
    pip install edge-tts

常用中文语音:
    - zh-CN-XiaoxiaoNeural  女声, 自然 (推荐)
    - zh-CN-YunxiNeural     男声, 专业
    - zh-CN-YunyeNeural     男声, 亲切
    - zh-CN-XiaoyiNeural    女声, 情感丰富
    - zh-CN-YunjianNeural   男声, 新闻播报
"""

import asyncio
import io
import time
import uuid
from typing import Iterator, Optional

from src.tts.base import BaseTTS, TTSResult
from src.utils.logger import get_logger

logger = get_logger(__name__)


class EdgeTTS(BaseTTS):
    """
    Microsoft Edge TTS

    使用示例:
        tts = EdgeTTS(voice="zh-CN-XiaoxiaoNeural")
        result = tts.synthesize("你好世界")
        with open("output.mp3", "wb") as f:
            f.write(result.audio_data)
    """

    def __init__(
        self,
        voice: str = "zh-CN-XiaoxiaoNeural",
        rate: str = "+0%",
        pitch: str = "+0Hz",
        proxy: Optional[str] = None,
    ):
        """
        Args:
            voice: 语音名称
            rate: 语速 (-50% ~ +100%)
            pitch: 音调
            proxy: HTTP 代理
        """
        self._voice = voice
        self._rate = rate
        self._pitch = pitch
        self._proxy = proxy
        self._available = True  # 先假定可用

    def synthesize(self, text: str) -> TTSResult:
        """
        合成语音 (同步包装)

        内部使用 asyncio, 对外提供同步接口
        """
        start_time = time.time()

        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # 如果已在事件循环中, 使用 nest_asyncio
                import nest_asyncio
                nest_asyncio.apply()
                audio_data = loop.run_until_complete(self._synthesize_async(text))
            else:
                audio_data = asyncio.run(self._synthesize_async(text))
        except RuntimeError:
            audio_data = asyncio.run(self._synthesize_async(text))

        latency_ms = (time.time() - start_time) * 1000

        logger.debug(f"Edge TTS 合成完成: \"{text[:50]}...\" ({latency_ms:.0f}ms, "
                      f"{len(audio_data)} bytes)")

        return TTSResult(
            audio_data=audio_data,
            format="mp3",
            sample_rate=24000,
            latency_ms=latency_ms,
        )

    async def _synthesize_async(self, text: str) -> bytes:
        """异步执行 TTS 合成"""
        import edge_tts

        communicate = edge_tts.Communicate(
            text=text,
            voice=self._voice,
            rate=self._rate,
            pitch=self._pitch,
            proxy=self._proxy,
        )

        # 收集音频数据
        audio_chunks = []
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                audio_chunks.append(chunk["data"])

        return b"".join(audio_chunks)

    def synthesize_stream(self, text: str) -> Iterator[bytes]:
        """
        流式语音合成

        注意: edge-tts 实际上是先生成完整音频再返回,
        流式 API 的效果有限, 但保留了接口以便未来升级
        """
        result = self.synthesize(text)
        # 按块返回 (10KB chunks)
        chunk_size = 10240
        for i in range(0, len(result.audio_data), chunk_size):
            yield result.audio_data[i:i + chunk_size]

    def synthesize_to_file(self, text: str, output_path: str) -> None:
        """合成并保存到文件"""
        result = self.synthesize(text)
        with open(output_path, "wb") as f:
            f.write(result.audio_data)
        logger.info(f"TTS 音频已保存: {output_path}")

    @property
    def is_available(self) -> bool:
        """检查 Edge TTS 是否可用"""
        if not self._available:
            return False
        try:
            # 快速测试
            import edge_tts
            return True
        except ImportError:
            logger.warning("edge_tts 未安装")
            return False

    @property
    def voice_name(self) -> str:
        return self._voice

    @staticmethod
    def list_voices() -> list:
        """列出所有可用的中文语音"""
        try:
            import edge_tts

            async def _list():
                voices = await edge_tts.list_voices()
                zh_voices = [
                    {
                        "name": v["ShortName"],
                        "locale": v["Locale"],
                        "gender": v.get("Gender", "Unknown"),
                        "friendly_name": v.get("FriendlyName", ""),
                    }
                    for v in voices
                    if v["Locale"].startswith("zh-")
                ]
                return zh_voices

            return asyncio.run(_list())
        except Exception as e:
            logger.error(f"获取语音列表失败: {e}")
            return []
