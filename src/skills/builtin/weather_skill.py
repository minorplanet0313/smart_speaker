"""
天气查询技能 (可选)

需要和风天气 API Key (免费)
注册: https://dev.qweather.com/

支持:
- "今天天气怎么样?"
- "明天会下雨吗?"
- "北京天气怎么样?"
"""

import gzip
import json
from typing import Optional
from urllib.error import HTTPError
from urllib.parse import quote
from urllib.request import urlopen, Request

from src.skills.base import BaseSkill, SkillContext, SkillPriority, SkillResult
from src.utils.logger import get_logger

logger = get_logger(__name__)

# 和风天气城市 ID (内置覆盖主要城市，避免依赖已下线的免费 Geo API)
_CITY_IDS = {
    # 直辖市
    "北京": "101010100", "上海": "101020100", "天津": "101030100",
    "重庆": "101040100",
    # 省会
    "广州": "101280100", "深圳": "101280600", "杭州": "101210100",
    "成都": "101270100", "武汉": "101200100", "南京": "101190100",
    "西安": "101110100", "长沙": "101250100", "郑州": "101180100",
    "济南": "101120100", "青岛": "101120200", "沈阳": "101070100",
    "大连": "101070200", "哈尔滨": "101050100", "长春": "101060100",
    "福州": "101230100", "厦门": "101230200", "合肥": "101220100",
    "南昌": "101240100", "贵阳": "101260100", "昆明": "101290100",
    "南宁": "101300100", "海口": "101310100", "三亚": "101310200",
    "石家庄": "101090100", "太原": "101100100", "呼和浩特": "101080100",
    "兰州": "101160100", "西宁": "101150100", "银川": "101170100",
    "乌鲁木齐": "101130100", "拉萨": "101140100",
    # 常见地级市
    "苏州": "101190400", "无锡": "101190200", "常州": "101191100",
    "南通": "101190500", "徐州": "101190800", "温州": "101210700",
    "宁波": "101210400", "绍兴": "101210500", "嘉兴": "101210300",
    "东莞": "101281600", "佛山": "101280800", "珠海": "101280700",
    "中山": "101281700", "惠州": "101280300", "泉州": "101230500",
    "烟台": "101120500", "淄博": "101120300", "洛阳": "101180900",
    "开封": "101180800", "桂林": "101300500", "大理": "101290200",
    "丽江": "101291400", "秦皇岛": "101091100", "宜昌": "101200900",
    "襄阳": "101200200", "岳阳": "101251000", "株洲": "101250300",
    "咸阳": "101110200", "宝鸡": "101110900", "绵阳": "101270400",
    "唐山": "101090500", "保定": "101090200", "邯郸": "101091000",
    "大同": "101100200", "运城": "101100800", "包头": "101080200",
    "鄂尔多斯": "101080700", "吉林": "101060200", "齐齐哈尔": "101050200",
    "大庆": "101050900", "牡丹江": "101050300", "鞍山": "101070300",
    "抚顺": "101070400", "威海": "101121300", "日照": "101121500",
    "潍坊": "101120600", "临沂": "101120900", "泰安": "101120800",
    "芜湖": "101220300", "安庆": "101220600", "九江": "101240200",
    "景德镇": "101240800", "赣州": "101240700", "南平": "101230900",
    "漳州": "101230600", "龙岩": "101230700", "柳州": "101300300",
    "北海": "101301300", "遵义": "101260200", "安顺": "101260300",
    "攀枝花": "101270200", "德阳": "101272000", "宜宾": "101271100",
    "泸州": "101271000", "南充": "101270500", "乐山": "101271400",
    "曲靖": "101290400", "玉溪": "101290700", "延安": "101110300",
    "汉中": "101110800", "天水": "101160900", "酒泉": "101160800",
    "张掖": "101161300", "嘉峪关": "101161400", "金昌": "101160600",
    "武威": "101160500", "白银": "101161200", "临夏": "101161100",
    "甘南": "101050200", "吴忠": "101170300", "石嘴山": "101170200",
    "固原": "101170400", "中卫": "101170500", "哈密": "101131200",
    "吐鲁番": "101130500", "喀什": "101130900", "克拉玛依": "101130200",
    "昌吉": "101130400", "博尔塔拉": "101131600", "巴音郭楞": "101130600",
    "阿克苏": "101130800", "克孜勒苏柯尔克孜": "101130700",
    "伊犁": "101131000", "塔城": "101131100", "阿勒泰": "101131400",
    "石河子": "101130300", "阿拉尔": "101130700", "图木舒克": "101130800",
    "五家渠": "101131400", "北屯": "101130700", "铁门关": "101130800",
    "双河": "101130800", "可克达拉": "101130800", "昆玉": "101130800",
    # 港澳台
    "香港": "101320100", "澳门": "101330100", "台北": "101340100",
    "高雄": "101340200", "台中": "101340400",
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

    def __init__(self, api_key: str, default_city: str = "auto",
                 api_host: str = "devapi.qweather.com"):
        super().__init__()
        self.api_key = api_key
        self.default_city = default_city
        self.api_host = api_host.rstrip("/")

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

    def _decode_response(self, raw: bytes) -> dict:
        """解码 API 响应（自动处理 gzip 压缩）"""
        if raw[:2] == b"\x1f\x8b":  # gzip magic bytes
            raw = gzip.decompress(raw)
        return json.loads(raw.decode())

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

        # 和风天气 API: 3天预报
        url = (
            f"https://{self.api_host}/v7/weather/3d?"
            f"location={city_id}&key={self.api_key}"
        )
        req = Request(url, headers={"User-Agent": "SmartSpeaker/1.0"})

        try:
            with urlopen(req, timeout=5) as resp:
                data = self._decode_response(resp.read())

            if data.get("code") == "200":
                return data
            else:
                logger.warning(f"天气 API 返回错误: code={data.get('code')}, "
                               f"data={data}")
                return None
        except HTTPError as e:
            # 读取错误响应体以获取 QWeather 错误码
            try:
                error_data = self._decode_response(e.read())
                qweather_code = error_data.get("code", "unknown")
                logger.error(
                    f"天气 API HTTP {e.code}: QWeather code={qweather_code}, "
                    f"city={city}, url={url.split('&key=')[0]}"
                )
            except Exception:
                logger.error(
                    f"天气 API HTTP {e.code}: {e.reason}, city={city}"
                )
            return None
        except Exception as e:
            logger.error(f"天气请求异常: {type(e).__name__}: {e}")
            return None

    def _search_city(self, city_name: str) -> Optional[str]:
        """搜索城市 ID（新 API Host 失败时回退到 geoapi.qweather.com）"""
        hosts = [self.api_host]
        if self.api_host != "geoapi.qweather.com":
            hosts.append("geoapi.qweather.com")  # 旧公共 geo API 兜底

        for host in hosts:
            try:
                url = (
                    f"https://{host}/v2/city/lookup?"
                    f"location={quote(city_name)}&key={self.api_key}"
                )
                req = Request(url, headers={"User-Agent": "SmartSpeaker/1.0"})
                with urlopen(req, timeout=5) as resp:
                    data = self._decode_response(resp.read())
                if data.get("code") == "200" and data.get("location"):
                    city_id = data["location"][0]["id"]
                    logger.debug(f"城市搜索 OK ({host}): {city_name} → {city_id}")
                    return city_id
            except HTTPError as e:
                if e.code == 404 and host != hosts[-1]:
                    continue  # 尝试下一个 host
                try:
                    error_data = self._decode_response(e.read())
                    logger.error(
                        f"城市搜索 HTTP {e.code}: QWeather code={error_data.get('code')}"
                    )
                except Exception:
                    logger.error(f"城市搜索 HTTP {e.code}: {e.reason}")
                return None
            except Exception as e:
                if host != hosts[-1]:
                    continue
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
