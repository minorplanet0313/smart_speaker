"""
通用聊天技能 (LLM 兜底)

当没有其他技能匹配用户输入时, 将所有文本发送给 LLM
这是优先级最低的兜底技能
"""

from src.llm.base import BaseLLM
from src.skills.base import BaseSkill, SkillContext, SkillPriority, SkillResult
from src.utils.logger import get_logger

logger = get_logger(__name__)


class ChatSkill(BaseSkill):
    """
    通用聊天 — 将用户输入转发给 LLM

    这是全局兜底技能, 当其他技能都不匹配时使用。
    LLM 调用是异步的, 结果通过 EventBus 返回。
    """

    name = "chat"
    description = "通用AI聊天 (通过 DeepSeek LLM)"
    keywords = []  # 不设关键词, 作为兜底
    priority = SkillPriority.FALLBACK
    require_network = True

    def __init__(self, llm: BaseLLM):
        super().__init__()
        self.llm = llm

    def can_handle(self, text: str) -> bool:
        """ChatSkill 作为兜底, 总是返回 True"""
        return True

    def execute(
        self,
        text: str,
        context: SkillContext,
    ) -> SkillResult:
        """
        不在这里直接调用 LLM, 而是标记 needs_llm=True
        让 Engine 发 Publishing 事件, 由事件驱动的 LLM 处理
        这样 LLM 调用是异步的, 不会阻塞技能管理器
        """
        return SkillResult(
            success=True,
            response_text="",       # LLM 响应会通过 EventBus 异步返回
            needs_llm=True,         # 告知 Engine 需要 LLM 处理
            data={"user_text": text},
        )
