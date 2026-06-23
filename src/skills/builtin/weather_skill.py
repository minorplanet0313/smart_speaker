"""
天气查询技能 (可选)

需要和风天气 API Key (免费)
注册: https://dev.qweather.com/

支持:
- "今天天气怎么样?"
- "明天会下雨吗?"
- "北京天气怎么样?"
"""

import json
from typing import Optional
from urllib.request import urlopen, Request

from src.skills.base import BaseSkill, SkillContext, SkillPriority, SkillResult
from src.utils.logger import get_logger

logger = get_logger(__name__)

# 和风天气城市 ID (简化版)
_CITY_IDS = {
    "北京": "101010100",
    "上海": "101020100",
    "广州": "101280100",
    "深圳": "101280600",
    "杭州": "101210100",
    "成都": "101270100",
    "武汉": "101200100",
    "南京": "101190100",
    "西安": "101110100",
    "重庆": "101040100",
}


class WeatherSkill(BaseSkill):
    """天气查询 (和风天气 API)"""

    name = "weather"
    description = "查询天气情况"
    keywords = [
        "天气", "下雨", "下雪", "温度",
        "气温", "刮风", "雾霾", "空气质量",
    ]
    priority = SkillPriority.HIGH
    require_network = True

    def __init__(self, api_key: str, default_city: str = "auto"):
        super().__init__()
        self.api_key = api_key
        self.default_city = default_city

    def can_handle(self, text: str) -> bool:
        return any(kw in text for kw in self.keywords)

    def execute(
        self,
        text: str,
        context: SkillContext,
    ) -> SkillResult:
        # 提取城市
        city = self._extract_city(text) or self.default_city
        if city == "auto":
            city = "北京"  # 默认城市

        # 判断查询类型
        is_tomorrow = "明天" in text or "明日" in text

        try:
            weather_info = self._fetch_weather(city)
            if weather_info is None:
                return SkillResult(
                    success=False,
                    response_text="抱歉, 暂时获取不到天气信息",
                    error_message="API request failed",
                )

            if is_tomorrow:
                response = self._format_tomorrow(city, weather_info)
            else:
                response = self._format_today(city, weather_info)

            return SkillResult(
                success=True,
                response_text=response,
                data=weather_info,
            )
        except Exception as e:
            logger.error(f"天气查询失败: {e}")
            return SkillResult(
                success=False,
                response_text="抱歉, 天气查询出了点问题",
                error_message=str(e),
            )

    def _extract_city(self, text: str) -> Optional[str]:
        """从文本中提取城市名"""
        for city in _CITY_IDS:
            if city in text:
                return city
        return None

    def _fetch_weather(self, city: str) -> Optional[dict]:
        """获取天气数据 (和风天气 API)"""
        city_id = _CITY_IDS.get(city)
        if not city_id:
            # 尝试城市搜索
            city_id = self._search_city(city)
            if not city_id:
                return None

        # 和风天气免费 API: 3天预报
        url = (
            f"https://devapi.qweather.com/v7/weather/3d?"
            f"location={city_id}&key={self.api_key}"
        )
        req = Request(url, headers={"User-Agent": "SmartSpeaker/1.0"})

        with urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read().decode())

        if data.get("code") == "200":
            return data
        else:
            logger.warning(f"天气 API 返回错误: {data}")
            return None

    def _search_city(self, city_name: str) -> Optional[str]:
        """搜索城市 ID"""
        try:
            url = (
                f"https://geoapi.qweather.com/v2/city/lookup?"
                f"location={city_name}&key={self.api_key}"
            )
            req = Request(url, headers={"User-Agent": "SmartSpeaker/1.0"})
            with urlopen(req, timeout=5) as resp:
                data = json.loads(resp.read().decode())
            if data.get("code") == "200" and data.get("location"):
                return data["location"][0]["id"]
        except Exception as e:
            logger.error(f"城市搜索失败: {e}")
        return None

    def _format_today(self, city: str, data: dict) -> str:
        """格式化今天的天气"""
        today = data["daily"][0]
        return (
            f"{city}今天{today['textDay']}, "
            f"气温{today['tempMin']}到{today['tempMax']}度, "
            f"{today['windDirDay']}风{today['windScaleDay']}级"
        )

    def _format_tomorrow(self, city: str, data: dict) -> str:
        """格式化明天的天气"""
        tomorrow = data["daily"][1]
        return (
            f"{city}明天{tomorrow['textDay']}, "
            f"气温{tomorrow['tempMin']}到{tomorrow['tempMax']}度, "
            f"{tomorrow['windDirDay']}风{tomorrow['windScaleDay']}级"
        )
