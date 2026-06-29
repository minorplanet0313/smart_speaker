"""
技能框架 — 基础类和接口

技能是可插拔的"能力模块", 类似于小爱音箱的"技能"。
每个技能可以:
1. 声明自己能处理哪些意图
2. 执行具体的业务逻辑
3. 返回响应文本

设计模式:
- BaseSkill: 技能抽象基类
- SkillContext: 执行上下文
- SkillResult: 执行结果
"""

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional


class SkillPriority(Enum):
    """技能优先级"""
    HIGH = 100     # 高优先级 (特定命令, 如时间/天气)
    NORMAL = 50    # 普通优先级
    LOW = 10       # 低优先级 (兜底, 如通用聊天)
    FALLBACK = 0   # 仅作为最后兜底


@dataclass
class SkillContext:
    """技能执行上下文"""
    conversation_id: str = ""
    user_id: str = "default"
    # 可扩展字段
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SkillResult:
    """技能执行结果"""
    success: bool
    response_text: str = ""          # 要播放/显示的响应文本
    data: Any = None                 # 结构化数据
    error_message: str = ""          # 错误信息
    needs_llm: bool = False          # 是否需要进一步 LLM 处理
    metadata: Dict[str, Any] = field(default_factory=dict)


class BaseSkill(ABC):
    """
    技能抽象基类

    使用示例:
        class MySkill(BaseSkill):
            name = "my_skill"
            description = "我的自定义技能"
            keywords = ["关键词1", "关键词2"]
            priority = SkillPriority.NORMAL

            def can_handle(self, text: str) -> bool:
                return any(kw in text for kw in self.keywords)

            def execute(self, text: str, context: SkillContext) -> SkillResult:
                return SkillResult(success=True, response_text="处理完成")
    """

    # --- 子类必须设置 ---
    name: str = ""               # 技能名称 (唯一标识)
    description: str = ""        # 技能描述
    keywords: List[str] = []     # 触发关键词
    priority: SkillPriority = SkillPriority.NORMAL

    # --- 子类可选覆盖 ---
    require_network: bool = False  # 是否需要网络

    def matches_keywords(self, text: str) -> bool:
        """关键词子串匹配，排除常见误触发（如「时间旅行」）"""
        if not self.keywords:
            return False
        for kw in self.keywords:
            if kw not in text:
                continue
            # 时间类技能：「时间」需作为独立词出现
            if kw == "时间" and re.search(r'时间(旅行|机器|胶囊|线|轴|管理)', text):
                continue
            return True
        return False

    def can_handle(self, text: str) -> bool:
        """默认实现：按关键词匹配"""
        return self.matches_keywords(text)

    @abstractmethod
    def execute(
        self,
        text: str,
        context: SkillContext,
    ) -> SkillResult:
        """
        执行技能

        Args:
            text: 用户输入文本
            context: 执行上下文

        Returns:
            SkillResult
        """
        ...

    def get_response_text(self, result: SkillResult) -> str:
        """从结果中提取响应文本 (可被子类覆盖)"""
        return result.response_text

    def on_load(self) -> None:
        """技能加载时的初始化 (可选)"""
        pass

    def on_unload(self) -> None:
        """技能卸载时的清理 (可选)"""
        pass

    def __repr__(self) -> str:
        return f"<Skill name={self.name!r} priority={self.priority}>"
