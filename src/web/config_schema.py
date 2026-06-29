"""
配置 Schema 定义

定义所有可配置字段的元数据，供：
- 后端 validate_updates() 校验
- 前端表单渲染 (GET /api/config/schema)
"""

from dataclasses import dataclass, field
from typing import Any, List, Optional


@dataclass
class ConfigField:
    """单个配置字段的元数据"""
    key: str                    # dot-path, e.g. "audio.vad.threshold"
    label: str                  # 中文标签
    category: str               # 分类: 通用|音频|唤醒词|ASR|LLM|TTS|对话|技能|其他
    field_type: str             # string|number|boolean|enum|secret|multiline
    default: Any = None
    description: str = ""
    enum_values: Optional[List[str]] = None
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    step: Optional[float] = None
    restart_required: bool = True   # 大部分配置需要重启, 少数可热生效
    advanced: bool = False          # 折叠在"高级"中
    device_type: str = ""           # "input" | "output" (仅 device_select 类型)


# ================================================================
# Schema 定义 — 覆盖 config.yaml 所有叶子节点
# ================================================================

SCHEMA: List[ConfigField] = [
    # ---- 通用 ----
    ConfigField(key="general.name", label="音箱名称", category="通用",
                field_type="string", default="小智音箱", restart_required=False),
    ConfigField(key="general.wake_word", label="唤醒词文本", category="通用",
                field_type="string", default="小智小智", restart_required=False),
    ConfigField(key="general.language", label="语言", category="通用",
                field_type="enum", default="zh-CN", enum_values=["zh-CN", "en-US"]),
    ConfigField(key="general.log_level", label="日志级别", category="通用",
                field_type="enum", default="INFO",
                enum_values=["DEBUG", "INFO", "WARNING", "ERROR"],
                restart_required=False),
    ConfigField(key="general.data_dir", label="数据目录", category="通用",
                field_type="string", default="./data", advanced=True,
                restart_required=False),

    # ---- 音频 ----
    ConfigField(key="audio.sample_rate", label="采样率 (Hz)", category="音频",
                field_type="enum", default=16000,
                enum_values=["8000", "16000", "22050", "44100"],
                restart_required=False),
    ConfigField(key="audio.channels", label="声道数", category="音频",
                field_type="enum", default=1, enum_values=["1", "2"],
                restart_required=False),
    ConfigField(key="audio.chunk_size", label="音频块大小 (帧)", category="音频",
                field_type="number", default=1024, min_value=256, max_value=4096,
                step=256, advanced=True),
    ConfigField(key="audio.format", label="音频格式", category="音频",
                field_type="enum", default="int16", enum_values=["int16", "float32"],
                advanced=True),
    ConfigField(key="audio.device.microphone", label="麦克风设备", category="音频",
                field_type="device_select", default="", device_type="input",
                description="留空使用系统默认"),
    ConfigField(key="audio.device.speaker", label="扬声器设备", category="音频",
                field_type="device_select", default="", device_type="output",
                description="留空使用系统默认"),

    # VAD
    ConfigField(key="audio.vad.enabled", label="VAD 启用", category="音频",
                field_type="boolean", default=True, restart_required=False),
    ConfigField(key="audio.vad.threshold", label="语音检测阈值", category="音频",
                field_type="number", default=0.3, min_value=0, max_value=1, step=0.05,
                description="越低越灵敏，轻声也能检测",
                restart_required=False),
    ConfigField(key="audio.vad.min_speech_duration_ms", label="最短语音 (ms)", category="音频",
                field_type="number", default=100, min_value=50, max_value=1000, step=50,
                advanced=True, restart_required=False),
    ConfigField(key="audio.vad.min_silence_duration_ms", label="静音判定 (ms)", category="音频",
                field_type="number", default=400, min_value=100, max_value=2000, step=50,
                description="越小响应越快，但容易截断语音",
                restart_required=False),
    ConfigField(key="audio.vad.max_speech_duration_ms", label="最长录音 (ms)", category="音频",
                field_type="number", default=15000, min_value=1000, max_value=60000, step=1000,
                advanced=True),
    ConfigField(key="audio.vad.speech_pad_ms", label="语音填充 (ms)", category="音频",
                field_type="number", default=300, min_value=0, max_value=1000, step=50,
                advanced=True),

    # ---- 唤醒词 ----
    ConfigField(key="wake_word.enabled", label="唤醒词启用", category="唤醒词",
                field_type="boolean", default=True),
    ConfigField(key="wake_word.engine", label="唤醒词引擎", category="唤醒词",
                field_type="enum", default="openwakeword", enum_values=["openwakeword"]),
    ConfigField(key="wake_word.model_path", label="模型路径", category="唤醒词",
                field_type="string", default="./models/openwakeword/xiao_zhi.onnx"),
    ConfigField(key="wake_word.threshold", label="检测阈值", category="唤醒词",
                field_type="number", default=0.5, min_value=0, max_value=1, step=0.05,
                description="越低越容易误触发，越高越难唤醒"),
    ConfigField(key="wake_word.cooldown_ms", label="冷却时间 (ms)", category="唤醒词",
                field_type="number", default=2000, min_value=500, max_value=10000, step=500,
                description="触发后多少毫秒内不重复检测",
                restart_required=False),
    ConfigField(key="wake_word.inference_framework", label="推理框架", category="唤醒词",
                field_type="enum", default="onnx", enum_values=["onnx", "tflite"],
                advanced=True),

    # ---- ASR ----
    ConfigField(key="asr.primary", label="主 ASR 引擎", category="ASR",
                field_type="enum", default="vosk",
                enum_values=["vosk", "sherpa", "cloud"],
                description="sherpa-onnx 推荐 (14MB, 高准确率)"),
    ConfigField(key="asr.fallback", label="降级引擎", category="ASR",
                field_type="enum", default="cloud", enum_values=["cloud", "none"],
                description="主引擎失败时的备用方案"),
    ConfigField(key="asr.incremental", label="增量识别", category="ASR",
                field_type="boolean", default=True,
                description="边听边识别，降低感知延迟 ~500ms"),
    ConfigField(key="asr.preprocess", label="音频预处理", category="ASR",
                field_type="boolean", default=True,
                description="去噪+归一化，提升识别准确率",
                restart_required=False),
    ConfigField(key="asr.vosk.model_path", label="Vosk 模型路径", category="ASR",
                field_type="string", default="./models/vosk-model-small-cn-0.22",
                advanced=True),
    ConfigField(key="asr.sherpa.model_path", label="Sherpa 模型路径", category="ASR",
                field_type="string",
                default="./models/sherpa-onnx-streaming-zipformer-zh-14M-2023-02-23",
                advanced=True),
    ConfigField(key="asr.sherpa.num_threads", label="Sherpa 推理线程数", category="ASR",
                field_type="number", default=2, min_value=1, max_value=8, step=1,
                advanced=True, restart_required=False),
    ConfigField(key="asr.sherpa.decoding_method", label="Sherpa 解码方式", category="ASR",
                field_type="enum", default="greedy_search",
                enum_values=["greedy_search", "modified_beam_search"], advanced=True),
    ConfigField(key="asr.cloud.provider", label="云端 ASR 提供商", category="ASR",
                field_type="enum", default="baidu",
                enum_values=["baidu", "tencent", "aliyun"]),
    ConfigField(key="asr.cloud.baidu.app_id", label="百度 App ID", category="ASR",
                field_type="string", default="", advanced=True),
    ConfigField(key="asr.cloud.baidu.api_key", label="百度 API Key", category="ASR",
                field_type="secret", default="", advanced=True),
    ConfigField(key="asr.cloud.baidu.secret_key", label="百度 Secret Key", category="ASR",
                field_type="secret", default="", advanced=True),
    ConfigField(key="asr.cloud.tencent.secret_id", label="腾讯 Secret ID", category="ASR",
                field_type="secret", default="", advanced=True),
    ConfigField(key="asr.cloud.tencent.secret_key", label="腾讯 Secret Key", category="ASR",
                field_type="secret", default="", advanced=True),

    # ---- LLM ----
    ConfigField(key="llm.provider", label="LLM 提供商", category="LLM",
                field_type="enum", default="deepseek", enum_values=["deepseek"]),
    ConfigField(key="llm.model", label="模型名称", category="LLM",
                field_type="string", default="deepseek-chat",
                description="deepseek-chat 或 deepseek-reasoner"),
    ConfigField(key="llm.api_key", label="API Key", category="LLM",
                field_type="secret", default="",
                description="DeepSeek API Key, 注册: platform.deepseek.com"),
    ConfigField(key="llm.base_url", label="API 地址", category="LLM",
                field_type="string", default="https://api.deepseek.com", advanced=True),
    ConfigField(key="llm.max_tokens", label="最大 Token 数", category="LLM",
                field_type="number", default=1024, min_value=64, max_value=8192, step=64,
                restart_required=False),
    ConfigField(key="llm.temperature", label="温度", category="LLM",
                field_type="number", default=0.7, min_value=0, max_value=2, step=0.1,
                description="越高越随机，越低越确定",
                restart_required=False),
    ConfigField(key="llm.stream", label="流式输出", category="LLM",
                field_type="boolean", default=True,
                description="流式输出 → 分句 TTS，首响延迟 < 1s"),
    ConfigField(key="llm.timeout_seconds", label="请求超时 (秒)", category="LLM",
                field_type="number", default=30, min_value=5, max_value=120, step=5,
                advanced=True),
    ConfigField(key="llm.retry.max_retries", label="最大重试次数", category="LLM",
                field_type="number", default=3, min_value=0, max_value=10, step=1,
                advanced=True),
    ConfigField(key="llm.retry.backoff_base", label="退避基数", category="LLM",
                field_type="number", default=2, min_value=1, max_value=10, step=0.5,
                advanced=True),

    # ---- TTS ----
    ConfigField(key="tts.primary", label="主 TTS 引擎", category="TTS",
                field_type="enum", default="edge", enum_values=["edge", "piper"]),
    ConfigField(key="tts.fallback", label="降级引擎", category="TTS",
                field_type="enum", default="piper", enum_values=["piper", "none"]),
    ConfigField(key="tts.edge.voice", label="Edge 语音", category="TTS",
                field_type="string", default="zh-CN-liaoning-XiaobeiNeural",
                description="微软 Edge TTS 语音名称",
                restart_required=False),
    ConfigField(key="tts.edge.rate", label="语速", category="TTS",
                field_type="string", default="+0%",
                description="-50% ~ +100%",
                restart_required=False),
    ConfigField(key="tts.edge.pitch", label="音调", category="TTS",
                field_type="string", default="+0Hz", advanced=True,
                restart_required=False),
    ConfigField(key="tts.edge.proxy", label="Edge 代理", category="TTS",
                field_type="string", default="", advanced=True),
    ConfigField(key="tts.piper.model_path", label="Piper 模型路径", category="TTS",
                field_type="string", default="./models/piper-voices/zh_CN-huayan-medium.onnx",
                advanced=True),

    # ---- 对话 ----
    ConfigField(key="conversation.max_history_rounds", label="最大对话轮数", category="对话",
                field_type="number", default=20, min_value=1, max_value=100, step=1,
                description="保留多少轮对话上下文"),
    ConfigField(key="conversation.context_timeout_seconds", label="上下文超时 (秒)", category="对话",
                field_type="number", default=300, min_value=30, max_value=3600, step=30,
                description="超过此时间自动清除上下文"),
    ConfigField(key="conversation.system_prompt", label="系统提示词", category="对话",
                field_type="multiline", default="你是一个智能语音助手，名叫小智。",
                description="LLM 角色设定"),

    # ---- 技能 ----
    ConfigField(key="skills.weather.api_key", label="天气 API Key", category="技能",
                field_type="secret", default="",
                description="和风天气 Key, 注册: dev.qweather.com"),

    # ---- 调试 ----
    ConfigField(key="debug.save_audio", label="保存调试音频", category="其他",
                field_type="boolean", default=False,
                description="保存每次语音结束的 WAV 文件",
                restart_required=False),

    # ---- Web ----
    ConfigField(key="web.enabled", label="Web 面板启用", category="其他",
                field_type="boolean", default=False),
    ConfigField(key="web.host", label="Web 监听地址", category="其他",
                field_type="string", default="0.0.0.0", advanced=True),
    ConfigField(key="web.port", label="Web 端口", category="其他",
                field_type="number", default=8080, min_value=1024, max_value=65535, step=1,
                advanced=True),

    # ---- 性能 ----
    ConfigField(key="performance.onnx_threads", label="ONNX 推理线程数", category="其他",
                field_type="number", default=2, min_value=1, max_value=8, step=1,
                advanced=True),
    ConfigField(key="performance.audio_buffer_seconds", label="音频缓冲区 (秒)", category="其他",
                field_type="number", default=0.5, min_value=0.1, max_value=5, step=0.1,
                advanced=True),
    ConfigField(key="performance.gc_threshold", label="GC 回收阈值", category="其他",
                field_type="number", default=1000, min_value=100, max_value=10000, step=100,
                advanced=True),
]

# ================================================================
# 工具函数
# ================================================================

def get_schema_list() -> list:
    """将 schema 转为 JSON 可序列化的 dict 列表 (给前端渲染表单)"""
    result = []
    for f in SCHEMA:
        d = {
            "key": f.key,
            "label": f.label,
            "category": f.category,
            "type": f.field_type,
            "default": f.default,
            "description": f.description,
            "restart_required": f.restart_required,
            "advanced": f.advanced,
        }
        if f.enum_values is not None:
            d["enum_values"] = f.enum_values
        if f.min_value is not None:
            d["min_value"] = f.min_value
        if f.max_value is not None:
            d["max_value"] = f.max_value
        if f.step is not None:
            d["step"] = f.step
        if f.device_type:
            d["device_type"] = f.device_type
        result.append(d)
    return result


def get_categories() -> list:
    """返回有序的分类列表"""
    seen = []
    for f in SCHEMA:
        if f.category not in seen:
            seen.append(f.category)
    return seen


def validate_updates(updates: dict, prefix: str = "") -> list:
    """
    校验配置更新，返回错误字符串列表。
    空列表表示全部通过。

    校验规则:
    - 未知 key 警告但不报错 (允许新增字段)
    - 类型错误报错 (number 值必须是数字, boolean 必须是布尔)
    - 范围超限报错 (min/max for number)
    - enum 不在取值范围内报错
    """
    errors = []
    schema_index = {f.key: f for f in SCHEMA}

    def _validate_flat(upd: dict, pfx: str):
        for key, val in upd.items():
            full_key = f"{pfx}.{key}" if pfx else key
            if isinstance(val, dict):
                _validate_flat(val, full_key)
            else:
                field = schema_index.get(full_key)
                if field is None:
                    # 未知 key — 不阻止保存，但提示
                    continue

                # 类型校验
                if field.field_type == "number":
                    if not isinstance(val, (int, float)) or isinstance(val, bool):
                        errors.append(f"{full_key}: 需要数字类型")
                    elif field.min_value is not None and val < field.min_value:
                        errors.append(
                            f"{field.label}: 不能小于 {field.min_value}，当前 {val}")
                    elif field.max_value is not None and val > field.max_value:
                        errors.append(
                            f"{field.label}: 不能大于 {field.max_value}，当前 {val}")

                elif field.field_type == "boolean":
                    if not isinstance(val, bool):
                        errors.append(f"{field.label}: 需要布尔类型 (true/false)")

                elif field.field_type == "enum" and field.enum_values is not None:
                    # 枚举值比较转为字符串 (前端可能传数字)
                    if str(val) not in field.enum_values:
                        errors.append(
                            f"{field.label}: 无效值 '{val}'，可选: {', '.join(field.enum_values)}")

    _validate_flat(updates, prefix)
    return errors
