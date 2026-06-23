"""
技能管理器

职责:
1. 注册/注销技能
2. 按优先级查找能处理输入文本的技能
3. 执行技能并返回结果
4. 自动发现 (import src/skills/builtin/ 下的技能)
"""

from typing import Dict, List, Optional

from src.skills.base import BaseSkill, SkillContext, SkillResult, SkillPriority
from src.utils.logger import get_logger

logger = get_logger(__name__)


class SkillManager:
    """
    技能管理器

    技能匹配策略:
    1. 遍历所有已注册技能 (按优先级降序)
    2. 调用 can_handle(text) 检查每个技能
    3. 返回第一个匹配的技能的 execute() 结果
    4. 如果没有技能匹配, 返回 needs_llm=True 的结果 (由 ChatSkill 处理)
    """

    def __init__(self):
        self._skills: Dict[str, BaseSkill] = {}
        self._fallback_skill: Optional[BaseSkill] = None

    def register(self, skill: BaseSkill) -> None:
        """
        注册技能

        Args:
            skill: 技能实例
        """
        if skill.name in self._skills:
            logger.warning(f"技能 '{skill.name}' 已存在, 将被覆盖")

        self._skills[skill.name] = skill
        skill.on_load()

        # 标记兜底技能
        if skill.priority == SkillPriority.FALLBACK:
            self._fallback_skill = skill

        logger.info(f"技能已注册: {skill.name} (priority={skill.priority.name})")

    def unregister(self, skill_name: str) -> bool:
        """
        注销技能

        Returns:
            True 如果成功移除
        """
        if skill_name not in self._skills:
            logger.warning(f"技能 '{skill_name}' 未注册")
            return False

        skill = self._skills.pop(skill_name)
        skill.on_unload()

        if self._fallback_skill and self._fallback_skill.name == skill_name:
            self._fallback_skill = None

        logger.info(f"技能已注销: {skill_name}")
        return True

    def find_handler(self, text: str) -> Optional[BaseSkill]:
        """
        查找能处理该文本的最优先技能

        Args:
            text: 用户输入文本

        Returns:
            匹配的技能实例, 或 None
        """
        # 按优先级降序排列
        sorted_skills = sorted(
            self._skills.values(),
            key=lambda s: s.priority.value,
            reverse=True,
        )

        for skill in sorted_skills:
            try:
                if skill.can_handle(text):
                    logger.debug(f"技能匹配: {skill.name} ← \"{text}\"")
                    return skill
            except Exception as e:
                logger.error(f"技能 {skill.name}.can_handle() 异常: {e}")

        return None

    def execute(
        self,
        text: str,
        context: SkillContext,
    ) -> SkillResult:
        """
        查找并执行最佳技能

        Args:
            text: 用户输入文本
            context: 执行上下文

        Returns:
            SkillResult
        """
        skill = self.find_handler(text)

        if skill is None:
            # 没有找到匹配的技能, 使用兜底
            if self._fallback_skill:
                logger.info(f"无技能匹配, 使用兜底技能: {self._fallback_skill.name}")
                skill = self._fallback_skill
            else:
                logger.warning(f"无技能能处理: \"{text}\", 且无兜底技能")
                return SkillResult(
                    success=False,
                    response_text="抱歉, 我还不知道怎么处理这个请求",
                    error_message="No matching skill",
                )

        try:
            logger.info(f"执行技能: {skill.name} ← \"{text}\"")
            result = skill.execute(text, context)
            logger.debug(f"技能结果: success={result.success}, "
                         f"text=\"{result.response_text[:80]}\"")
            return result
        except Exception as e:
            logger.error(f"技能 {skill.name}.execute() 异常: {e}", exc_info=True)
            return SkillResult(
                success=False,
                response_text="抱歉, 出了点问题, 请再试一次",
                error_message=str(e),
            )

    def list_skills(self) -> List[dict]:
        """列出所有技能"""
        return [
            {
                "name": s.name,
                "description": s.description,
                "keywords": s.keywords,
                "priority": s.priority.name,
            }
            for s in sorted(
                self._skills.values(),
                key=lambda x: x.priority.value,
                reverse=True,
            )
        ]

    def clear(self) -> None:
        """清除所有技能"""
        for skill in list(self._skills.values()):
            skill.on_unload()
        self._skills.clear()
        self._fallback_skill = None
        logger.info("所有技能已清除")
