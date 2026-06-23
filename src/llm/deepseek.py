"""
DeepSeek API 客户端

使用 OpenAI 兼容的 SDK 调用 DeepSeek Chat API
- 支持流式和非流式输出
- 指数退避重试
- 超时保护

API 文档: https://platform.deepseek.com/api-docs
"""

import functools
import time
from typing import Dict, Iterator, List, Optional

from openai import OpenAI

from src.llm.base import BaseLLM, LLMResponse
from src.utils.logger import get_logger

logger = get_logger(__name__)


def retry_on_error(
    max_retries: int = 3,
    base_delay: float = 1.0,
    backoff_factor: float = 2.0,
):
    """指数退避重试装饰器"""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    if attempt < max_retries - 1:
                        delay = base_delay * (backoff_factor ** attempt)
                        logger.warning(
                            f"API 调用失败 (尝试 {attempt+1}/{max_retries}): {e}, "
                            f"{delay:.1f}s 后重试..."
                        )
                        time.sleep(delay)
                    else:
                        logger.error(f"API 调用全部失败 (共{max_retries}次): {e}")
            raise last_exception
        return wrapper
    return decorator


class DeepSeekLLM(BaseLLM):
    """
    DeepSeek Chat API 客户端

    使用示例:
        llm = DeepSeekLLM(api_key="sk-xxx")
        response = llm.chat([{"role": "user", "content": "你好"}])
        print(response.text)

        # 流式
        for chunk in llm.chat_stream([{"role": "user", "content": "讲个故事"}]):
            print(chunk, end="", flush=True)
    """

    def __init__(
        self,
        api_key: str,
        model: str = "deepseek-chat",
        base_url: str = "https://api.deepseek.com",
        system_prompt: str = "",
        temperature: float = 0.7,
        max_tokens: int = 1024,
        timeout: int = 30,
    ):
        """
        Args:
            api_key: DeepSeek API Key
            model: 模型名称 (deepseek-chat, deepseek-reasoner)
            base_url: API 地址
            system_prompt: 系统提示词
            temperature: 温度参数 [0, 2]
            max_tokens: 最大生成 token 数
            timeout: 请求超时 (秒)
        """
        self.model = model
        self.system_prompt = system_prompt
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout

        self._client = OpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=timeout,
        )

        logger.info(f"DeepSeek 客户端初始化: model={model}, base_url={base_url}")

    @retry_on_error(max_retries=3, base_delay=1.0)
    def chat(
        self,
        messages: List[Dict[str, str]],
        stream: bool = False,
        **kwargs,
    ) -> LLMResponse:
        """
        发送消息并获取回复

        Args:
            messages: 消息列表
            stream: 是否流式返回 (注意: 非流式才一次返回完整结果)
            **kwargs: 覆盖默认参数

        Returns:
            LLMResponse
        """
        start_time = time.time()

        # 准备消息
        full_messages = self._prepare_messages(messages)

        # 合并参数
        params = {
            "model": self.model,
            "messages": full_messages,
            "temperature": kwargs.get("temperature", self.temperature),
            "max_tokens": kwargs.get("max_tokens", self.max_tokens),
        }

        if stream:
            # 流式: 收集所有块
            chunks = []
            response = self._client.chat.completions.create(
                **params, stream=True
            )
            for chunk in response:
                if chunk.choices and chunk.choices[0].delta.content:
                    chunks.append(chunk.choices[0].delta.content)

            text = "".join(chunks)
        else:
            # 非流式
            response = self._client.chat.completions.create(
                **params, stream=False
            )
            text = response.choices[0].message.content or ""

        latency_ms = (time.time() - start_time) * 1000
        tokens = response.usage.total_tokens if hasattr(response, 'usage') else 0

        logger.debug(f"DeepSeek 响应: \"{text[:80]}...\" ({latency_ms:.0f}ms, {tokens}tokens)")

        return LLMResponse(
            text=text,
            tokens_used=tokens,
            latency_ms=latency_ms,
        )

    @retry_on_error(max_retries=2, base_delay=1.0)
    def chat_stream(
        self,
        messages: List[Dict[str, str]],
        **kwargs,
    ) -> Iterator[str]:
        """
        流式聊天

        Yields:
            文本增量 (delta)
        """
        full_messages = self._prepare_messages(messages)

        params = {
            "model": self.model,
            "messages": full_messages,
            "temperature": kwargs.get("temperature", self.temperature),
            "max_tokens": kwargs.get("max_tokens", self.max_tokens),
            "stream": True,
        }

        response = self._client.chat.completions.create(**params)

        for chunk in response:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

    def _prepare_messages(
        self,
        messages: List[Dict[str, str]],
    ) -> List[Dict[str, str]]:
        """在消息列表开头添加 system prompt (如果还没有)"""
        if not messages:
            return messages

        has_system = any(m.get("role") == "system" for m in messages)
        if not has_system and self.system_prompt:
            return [{"role": "system", "content": self.system_prompt}] + list(messages)
        return list(messages)

    @property
    def is_available(self) -> bool:
        """检查 API 是否可用 (简单 ping)"""
        try:
            # 快速检查
            self._client.models.list()
            return True
        except Exception:
            return False
