"""LLM 流式输出分句工具"""

import re

_SENTENCE_END = re.compile(r'([。！？!?；;…\n])')


def extract_complete_sentences(buffer: str) -> tuple:
    """
    从缓冲区提取已完成的句子。

    Returns:
        (完整句子列表, 剩余未完成文本)
    """
    if not buffer:
        return [], ""

    sentences: list[str] = []
    parts = _SENTENCE_END.split(buffer)
    if len(parts) == 1:
        return [], buffer

    i = 0
    while i < len(parts) - 1:
        sentence = parts[i] + parts[i + 1]
        stripped = sentence.strip()
        if stripped:
            sentences.append(stripped)
        i += 2

    remainder = parts[-1] if len(parts) % 2 == 1 else ""
    return sentences, remainder
