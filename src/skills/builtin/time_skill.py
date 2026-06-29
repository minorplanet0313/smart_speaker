"""
时间查询技能

支持:
- "现在几点了?"
- "今天是几号?"
- "今天星期几?"
- "现在是什么时间?"
"""

from datetime import datetime

from src.skills.base import BaseSkill, SkillContext, SkillPriority, SkillResult

# 中文数字映射
_WEEKDAY_NAMES = {
    0: "星期一",
    1: "星期二",
    2: "星期三",
    3: "星期四",
    4: "星期五",
    5: "星期六",
    6: "星期日",
}


class TimeSkill(BaseSkill):
    """时间查询"""

    name = "time"
    description = "查询当前时间和日期"
    keywords = [
        "几点", "时间", "几号", "日期",
        "星期几", "周几", "礼拜几",
        "今天星期", "今天周",
    ]
    priority = SkillPriority.HIGH  # 高优先级, 精确匹配
    require_network = False

    def execute(
        self,
        text: str,
        context: SkillContext,
    ) -> SkillResult:
        now = datetime.now()

        # 判断具体问什么
        if "星期" in text or "周几" in text or "礼拜" in text:
            weekday = _WEEKDAY_NAMES[now.weekday()]
            response = f"今天是{weekday}"
        elif "几号" in text or "日期" in text:
            response = f"今天是{now.year}年{now.month}月{now.day}日"
        else:
            # 默认返回完整时间
            response = (
                f"现在是{now.year}年{now.month}月{now.day}日 "
                f"{_WEEKDAY_NAMES[now.weekday()]} "
                f"{now.hour}点{now.minute}分"
            )

        return SkillResult(
            success=True,
            response_text=response,
            data={"datetime": now.isoformat()},
        )
