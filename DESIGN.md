# 🎙️ Smart Speaker — 智能语音助手完整技术方案

> 基于 Raspberry Pi 3B+ 的类"小爱音箱"语音交互系统

---

## 目录

1. [系统整体架构](#1-系统整体架构)
2. [软件技术栈](#2-软件技术栈)
3. [Raspberry Pi 系统环境建议](#3-raspberry-pi-系统环境建议)
4. [项目目录结构](#4-项目目录结构)
5. [Python 模块划分](#5-python-模块划分)
6. [各模块接口设计](#6-各模块接口设计)
7. [数据流和交互流程](#7-数据流和交互流程)
8. [开机自启动方案](#8-开机自启动方案)
9. [异常处理方案](#9-异常处理方案)
10. [性能优化方案](#10-性能优化方案)
11. [后续扩展方案](#11-后续扩展方案)

---

## 1. 系统整体架构

### 1.1 架构图

```
┌─────────────────────────────────────────────────────────────────────┐
│                        Smart Speaker System                         │
│                                                                     │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐     │
│  │ USB Mic  │───▶│  Audio   │───▶│  Wake    │───▶│   VAD    │     │
│  │          │    │ Capture  │    │  Word    │    │ (Silero) │     │
│  └──────────┘    └──────────┘    │ Detector │    └────┬─────┘     │
│                                  │(openWake │         │            │
│                                  │  Word)   │         │            │
│                                  └────┬─────┘         │            │
│                                       │               │            │
│                                       ▼               ▼            │
│                              ┌─────────────────────────────┐       │
│                              │       Event Bus              │       │
│                              │   (Observer Pattern)         │       │
│                              └──────────────┬──────────────┘       │
│                                             │                      │
│                    ┌────────────────────────┼──────────────┐       │
│                    │                        │              │       │
│                    ▼                        ▼              ▼       │
│             ┌──────────┐           ┌──────────┐    ┌──────────┐   │
│             │   ASR    │           │   LLM    │    │   TTS    │   │
│             │ (Vosk/   │──────────▶│(DeepSeek)│───▶│(edge-tts │   │
│             │  Cloud)  │   text    │          │text│ /Piper)  │   │
│             └──────────┘           └──────────┘    └────┬─────┘   │
│                                                         │         │
│                                                         ▼         │
│                                                  ┌──────────┐     │
│  ┌──────────┐                                    │  Audio   │     │
│  │ USB Spkr │◀───────────────────────────────────│  Player  │     │
│  └──────────┘                                    └──────────┘     │
│                                                                     │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │                     GPIO (Optional)                           │  │
│  │   ┌────────┐  ┌──────────┐  ┌──────────┐                    │  │
│  │   │  LED   │  │  Button  │  │  Others  │                    │  │
│  │   │(status)│  │ (mute/   │  │ (sensors)│                    │  │
│  │   │        │  │  trigger) │  │          │                    │  │
│  │   └────────┘  └──────────┘  └──────────┘                    │  │
│  └──────────────────────────────────────────────────────────────┘  │
│                                                                     │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │                    Skills Framework                           │  │
│  │   ┌────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐     │  │
│  │   │  Chat  │  │  Weather │  │   Time   │  │   HASS   │     │  │
│  │   │ (LLM)  │  │  Skill   │  │  Skill   │  │  (未来)   │     │  │
│  │   └────────┘  └──────────┘  └──────────┘  └──────────┘     │  │
│  └──────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
```

### 1.2 模块关系

```
main.py
  └── Engine (core/engine.py)
        ├── EventBus (core/event_bus.py)
        ├── StateMachine (core/state_machine.py)
        ├── AudioCapture (audio/capture.py)
        ├── AudioPlayer (audio/player.py)
        ├── VAD (audio/vad.py)
        ├── WakeWordDetector (wake_word/detector.py)
        ├── ASR (asr/base.py → vosk_asr.py / cloud_asr.py)
        ├── LLM (llm/base.py → deepseek.py)
        ├── TTS (tts/base.py → edge_tts.py / piper_tts.py)
        ├── SkillManager (skills/skill_manager.py)
        └── GPIOManager (gpio/led.py, gpio/button.py)
```

### 1.3 状态机

```
                    ┌──────────────────────────────────┐
                    │                                  │
                    ▼                                  │
              ┌─────────┐    wake word    ┌──────────┐ │
        ┌────▶│  IDLE   │───────────────▶│LISTENING │ │
        │     └─────────┘                └────┬─────┘ │
        │          ▲                          │       │
        │          │                   silence/VAD    │
        │          │                          │       │
        │          │                          ▼       │
        │     ┌─────────┐               ┌──────────┐ │
        │     │ SPEAKING│◀──────────────│ THINKING │ │
        │     └─────────┘   TTS done    └────┬─────┘ │
        │          │                          │       │
        │          │                          │       │
        └──────────┘    playback done    ASR→LLM→TTS │
                                           │         │
                                           └─────────┘
```

---

## 2. 软件技术栈

### 2.1 核心选型总览

| 层级 | 组件 | 选型 | License | 运行位置 | 原因 |
|------|------|------|---------|----------|------|
| **唤醒词** | openWakeWord | Apache 2.0 | 本地 | Pi3 单核可跑15-20个模型, 延迟~200ms, Python原生支持 |
| **VAD** | Silero VAD | MIT | 本地 | 模型仅2MB, 推理<10ms, ONNX Runtime加速 |
| **ASR(主)** | Vosk (small-cn-0.22) | Apache 2.0 | 本地 | 模型42MB, 内存<120MB, 中文准确率~85% |
| **ASR(备)** | 阿里云/百度 ASR API | 商业 | 云端 | 准确率>95%, 作为长句/复杂场景补充 |
| **LLM** | DeepSeek API | 商业 | 云端 | 用户指定, 中文能力强, 价格低 |
| **TTS(主)** | edge-tts | 免费 | 云端 | 中文质量优秀, 30+神经语音, 零成本 |
| **TTS(备)** | Piper TTS | MIT | 本地 | 离线可用, 中文女声, Pi3可运行 |
| **音频** | PyAudio + PulseAudio | MIT/GPL | 本地 | 稳定成熟, 支持USB设备热插拔 |
| **GPIO** | RPi.GPIO / gpiozero | MIT | 本地 | LED状态指示, 物理按键 |

### 2.2 唤醒词 — openWakeWord

**选择原因:**
- Apache 2.0 开源协议, 完全免费
- 在 Pi 3 单核上可同时运行 15-20 个模型
- 支持自定义唤醒词 (如 "小智小智")
- Python 原生 `pip install openwakeword`
- Home Assistant 官方推荐, 生态成熟

**安装:**
```bash
pip install openwakeword
```

**关键API:**
```python
from openwakeword import Model
oww = Model(wakeword_models=["path/to/custom_model.onnx"])
scores = oww.predict(audio_frame)  # audio_frame: (N, 1280) @16kHz
if scores["custom_model"] > threshold:
    trigger_wake()
```

### 2.3 VAD — Silero VAD

**选择原因:**
- MIT 协议
- 模型仅 2MB, ONNX 格式
- 推理时间 <10ms (Pi 4), Pi 3 可接受
- 支持 8000+ 语言
- 精确检测语音起止点

**安装:**
```bash
pip install silero-vad onnxruntime
```

### 2.4 ASR — Vosk (本地) + 云端API (备选)

**Vosk 本地识别:**
- 模型 `vosk-model-small-cn-0.22` 仅 42MB
- 加载后内存占用 ~80-120MB
- 延迟 150-400ms (Pi 3)
- 中文短句准确率 ~85%
- 完全离线, 隐私保护

**云端备选:**
- 阿里云一句话识别: ¥0.0008/次
- 百度短语音识别: 免费额度 5万次/天
- 用于复杂长句、数字识别等场景

**Vosk安装:**
```bash
pip install vosk
# 下载模型
wget https://alphacephei.com/vosk/models/vosk-model-small-cn-0.22.zip
unzip vosk-model-small-cn-0.22.zip -d models/
```

### 2.5 LLM — DeepSeek

使用 DeepSeek Chat API, 中文理解和生成能力优秀。

**安装:**
```bash
pip install openai  # DeepSeek 兼容 OpenAI SDK
```

**API 调用:**
```python
from openai import OpenAI
client = OpenAI(
    api_key="your-deepseek-api-key",
    base_url="https://api.deepseek.com"
)
response = client.chat.completions.create(
    model="deepseek-chat",
    messages=[{"role": "user", "content": "你好"}],
    stream=True  # 流式输出降低首字延迟
)
```

### 2.6 TTS — edge-tts (主力) + Piper (离线)

**edge-tts:**
- 微软 Edge 神经语音, 完全免费
- 中文语音 30+ 种 (云溪、晓晓、云野等)
- 支持 SSML (语速、音调控制)
- 延迟 ~500ms-1.5s

**Piper TTS (离线备份):**
- MIT 协议, 本地运行
- 中文语音可用
- 延迟 ~300ms-800ms (Pi 3)

**安装:**
```bash
pip install edge-tts
pip install piper-tts
# 下载中文语音模型
wget https://huggingface.co/rhasspy/piper-voices/resolve/main/zh/zh_CN/huayan/medium/zh_CN-huayan-medium.onnx
```

### 2.7 音频管理

- **PulseAudio**: 管理 USB 音频设备, 支持多设备路由
- **PyAudio**: Python 音频捕获/播放
- 采样率统一为 **16kHz, 16bit, 单声道** (所有模型的统一标准)

---

## 3. Raspberry Pi 系统环境建议

### 3.1 操作系统

```bash
# 推荐 Raspberry Pi OS Lite (Bookworm, 64-bit)
# 无桌面环境, 节省 ~200MB RAM
# 下载: https://www.raspberrypi.com/software/operating-systems/
```

### 3.2 系统配置

```bash
# 1. 扩大 swap (重要! 编译和加载模型时需要)
sudo dphys-swapfile swapoff
sudo sed -i 's/CONF_SWAPSIZE=.*/CONF_SWAPSIZE=2048/' /etc/dphys-swapfile
sudo dphys-swapfile setup
sudo dphys-swapfile swapon

# 2. CPU 性能模式
echo "performance" | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor

# 3. 安装系统依赖
sudo apt update
sudo apt install -y \
    python3 python3-pip python3-venv \
    portaudio19-dev python3-pyaudio \
    pulseaudio pulseaudio-utils \
    libsndfile1 libasound2-dev \
    libatlas-base-dev \
    git wget curl

# 4. 配置 PulseAudio (确保 USB 音频设备优先级)
# 编辑 /etc/pulse/default.pa, 添加:
# load-module module-switch-on-connect

# 5. 禁用不必要的服务
sudo systemctl disable bluetooth
sudo systemctl disable avahi-daemon

# 6. 设置音频设备
# 用 pactl list short sources 查看麦克风
# 用 pactl list short sinks 查看扬声器
```

### 3.3 Python 环境

```bash
# 创建虚拟环境
python3 -m venv ~/smart_speaker_venv
source ~/smart_speaker_venv/bin/activate

# 使用清华源加速
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
```

---

## 4. 项目目录结构

```
smart_speaker/
├── README.md                         # 项目说明
├── DESIGN.md                         # 本设计文档
├── requirements.txt                  # Python 依赖
├── setup.py                          # 安装配置 (可选)
│
├── config/
│   ├── config.yaml                   # 主配置文件
│   └── prompts.yaml                  # LLM 系统提示词
│
├── src/
│   ├── __init__.py
│   ├── main.py                       # 程序入口
│   │
│   ├── core/                         # 核心框架
│   │   ├── __init__.py
│   │   ├── engine.py                 # 主引擎, 协调所有模块
│   │   ├── state_machine.py          # 状态机 (IDLE/LISTENING/THINKING/SPEAKING)
│   │   └── event_bus.py             # 事件总线 (发布/订阅)
│   │
│   ├── audio/                        # 音频处理
│   │   ├── __init__.py
│   │   ├── capture.py               # 麦克风音频捕获
│   │   ├── player.py                # 扬声器音频播放
│   │   ├── vad.py                   # 语音活动检测 (Silero VAD)
│   │   └── utils.py                 # 音频工具 (重采样、格式转换)
│   │
│   ├── wake_word/                    # 唤醒词检测
│   │   ├── __init__.py
│   │   └── detector.py              # openWakeWord 封装
│   │
│   ├── asr/                          # 语音识别
│   │   ├── __init__.py
│   │   ├── base.py                  # ASR 抽象接口
│   │   ├── vosk_asr.py             # Vosk 本地 ASR 实现
│   │   └── cloud_asr.py            # 云端 ASR 实现 (阿里/百度)
│   │
│   ├── llm/                          # 大语言模型
│   │   ├── __init__.py
│   │   ├── base.py                  # LLM 抽象接口
│   │   ├── deepseek.py             # DeepSeek API 客户端
│   │   └── context.py              # 对话上下文管理
│   │
│   ├── tts/                          # 语音合成
│   │   ├── __init__.py
│   │   ├── base.py                  # TTS 抽象接口
│   │   ├── edge_tts.py             # Edge TTS 实现
│   │   └── piper_tts.py            # Piper 本地 TTS 实现
│   │
│   ├── skills/                       # 技能框架
│   │   ├── __init__.py
│   │   ├── base.py                  # 基础技能类
│   │   ├── skill_manager.py         # 技能注册与调度
│   │   └── builtin/
│   │       ├── __init__.py
│   │       ├── chat_skill.py        # 通用聊天 (走 LLM)
│   │       ├── time_skill.py        # 时间查询
│   │       └── weather_skill.py     # 天气查询
│   │
│   ├── gpio/                         # GPIO 外设 (可选)
│   │   ├── __init__.py
│   │   ├── led.py                   # LED 状态指示灯
│   │   └── button.py                # 物理按键
│   │
│   └── utils/                        # 工具模块
│       ├── __init__.py
│       ├── config.py                # 配置加载
│       └── logger.py                # 日志配置
│
├── scripts/
│   ├── install.sh                    # 一键安装脚本
│   ├── start.sh                      # 启动脚本
│   ├── download_models.sh            # 模型下载脚本
│   └── service/
│       └── smart-speaker.service     # systemd 服务文件
│
├── models/                           # 本地模型存储 (.gitignore)
│   ├── vosk-model-small-cn-0.22/
│   ├── openwakeword/
│   └── piper-voices/
│
└── tests/                            # 测试
    ├── __init__.py
    ├── test_asr.py
    ├── test_llm.py
    └── test_tts.py
```

---

## 5. Python 模块划分

### 5.1 模块职责

| 模块 | 文件 | 职责 | 依赖 |
|------|------|------|------|
| **入口** | `main.py` | 解析参数, 初始化引擎, 启动主循环 | core |
| **引擎** | `core/engine.py` | 创建所有模块, 协调生命周期, 处理事件 | all |
| **事件总线** | `core/event_bus.py` | 模块间解耦通信 (发布/订阅) | 无 |
| **状态机** | `core/state_machine.py` | 管理交互状态转换 | event_bus |
| **音频捕获** | `audio/capture.py` | 麦克风流式采集, 16kHz 单声道 | utils |
| **音频播放** | `audio/player.py` | 扬声器播放, 支持流式 | utils |
| **VAD** | `audio/vad.py` | 检测语音起止点 | 无 |
| **唤醒词** | `wake_word/detector.py` | 实时监听唤醒词 | 无 |
| **ASR接口** | `asr/base.py` | 定义 ASR 抽象接口 | 无 |
| **ASR实现** | `asr/vosk_asr.py` | Vosk 离线识别 | base |
| **ASR实现** | `asr/cloud_asr.py` | 云端 ASR (阿里/百度) | base |
| **LLM接口** | `llm/base.py` | 定义 LLM 抽象接口 | 无 |
| **LLM实现** | `llm/deepseek.py` | DeepSeek API 调用 | base |
| **对话上下文** | `llm/context.py` | 多轮对话历史管理 | 无 |
| **TTS接口** | `tts/base.py` | 定义 TTS 抽象接口 | 无 |
| **TTS实现** | `tts/edge_tts.py` | Edge TTS 在线合成 | base |
| **TTS实现** | `tts/piper_tts.py` | Piper 本地合成 | base |
| **技能基类** | `skills/base.py` | 技能抽象基类 | 无 |
| **技能管理** | `skills/skill_manager.py` | 注册/匹配/执行技能 | base |
| **GPIO** | `gpio/led.py` | LED 状态指示 | 无 |
| **GPIO** | `gpio/button.py` | 物理按键输入 | 无 |
| **配置** | `utils/config.py` | 加载/合并 YAML 配置 | 无 |
| **日志** | `utils/logger.py` | 统一日志配置 | config |

---

## 6. 各模块接口设计

### 6.1 EventBus (事件总线)

```python
class EventBus:
    """发布/订阅事件总线, 解耦模块间通信"""
    
    # 标准事件类型
    class Event(Enum):
        WAKE_WORD_DETECTED = "wake_word_detected"
        SPEECH_START       = "speech_start"
        SPEECH_END         = "speech_end"
        ASR_RESULT         = "asr_result"          # data: {"text": str}
        ASR_ERROR          = "asr_error"
        LLM_RESPONSE       = "llm_response"         # data: {"text": str}
        LLM_STREAM_CHUNK   = "llm_stream_chunk"     # data: {"delta": str}
        LLM_ERROR          = "llm_error"
        TTS_AUDIO_READY    = "tts_audio_ready"      # data: {"audio": bytes, "format": str}
        TTS_DONE           = "tts_done"
        TTS_ERROR          = "tts_error"
        PLAYBACK_START     = "playback_start"
        PLAYBACK_DONE      = "playback_done"
        STATE_CHANGED      = "state_changed"        # data: {"from": str, "to": str}
        ERROR              = "error"                # data: {"source": str, "message": str}
        BUTTON_PRESSED     = "button_pressed"       # data: {"pin": int}
        SHUTDOWN           = "shutdown"

    def subscribe(self, event: Event, callback: Callable) -> None
    def unsubscribe(self, event: Event, callback: Callable) -> None
    def publish(self, event: Event, **data) -> None
```

### 6.2 StateMachine (状态机)

```python
class State(Enum):
    IDLE      = "idle"       # 等待唤醒
    LISTENING = "listening"  # 正在听用户说话
    THINKING  = "thinking"   # 处理中 (ASR+LLM+TTS)
    SPEAKING  = "speaking"   # 播放回复
    MUTED     = "muted"      # 静音模式
    ERROR     = "error"      # 错误状态

class StateMachine:
    @property
    def current_state(self) -> State
    
    def transition(self, to_state: State) -> bool  # 返回是否允许转换
    def can_transition(self, to_state: State) -> bool
    def on_state_changed(self, callback: Callable[[State, State], None])
```

**状态转换规则:**
```
IDLE      → LISTENING  (wake word detected)
LISTENING → THINKING   (speech ended / VAD silence)
LISTENING → IDLE       (timeout, no speech detected)
THINKING  → SPEAKING   (TTS audio ready)
THINKING  → ERROR      (ASR/LLM/TTS failed)
THINKING  → IDLE       (empty ASR result)
SPEAKING  → IDLE       (playback complete)
SPEAKING  → LISTENING  (wake word during playback, barge-in)
ANY       → MUTED      (button / command)
MUTED     → IDLE       (button / command)
ERROR     → IDLE       (auto-recovery)
```

### 6.3 AudioCapture (音频捕获)

```python
class AudioCapture:
    """
    基于 PyAudio 的流式音频捕获
    - 16kHz, 16bit, 单声道
    - 使用回调函数模式, 非阻塞
    - 音频帧通过回调函数传递给消费者
    """
    def __init__(self, config: dict)
    def start(self, callback: Callable[[np.ndarray], None]) -> None
    def stop(self) -> None
    @property
    def is_running(self) -> bool
    @property
    def sample_rate(self) -> int  # 16000
    
    # callback 参数: numpy array, shape=(N_samples,), dtype=float32, range=[-1, 1]
```

### 6.4 AudioPlayer (音频播放)

```python
class AudioPlayer:
    """音频播放器, 支持 WAV/MP3, 支持打断"""
    def play(self, audio_data: bytes, format: str = "mp3") -> None  # 阻塞播放
    def play_async(self, audio_data: bytes, format: str = "mp3") -> threading.Thread
    def stop(self) -> None  # 打断当前播放 (barge-in)
    def wait_until_done(self) -> None
    @property
    def is_playing(self) -> bool
```

### 6.5 VAD (语音活动检测)

```python
class VoiceActivityDetector:
    """
    Silero VAD 封装
    检测语音起始和结束
    """
    def __init__(self, 
                 threshold: float = 0.5,
                 min_speech_duration_ms: int = 250,
                 min_silence_duration_ms: int = 800,
                 speech_pad_ms: int = 200)
    
    def is_speech(self, audio_frame: np.ndarray) -> bool
    def process(self, audio_frame: np.ndarray) -> VADState  # SPEECH / SILENCE / START / END
    def reset(self) -> None

class VADState(Enum):
    SILENCE_START = auto()
    SPEECH_START  = auto()
    IN_SPEECH     = auto()
    SPEECH_END    = auto()
    IN_SILENCE    = auto()
```

### 6.6 WakeWordDetector (唤醒词)

```python
class WakeWordDetector:
    """
    openWakeWord 封装
    支持自定义唤醒词 (如 "小智小智")
    """
    def __init__(self, 
                 model_path: str,
                 threshold: float = 0.5,
                 inference_framework: str = "onnx")
    
    def detect(self, audio_frame: np.ndarray) -> float  # 返回置信度分数 [0, 1]
    def set_threshold(self, threshold: float)
    def on_detected(self, callback: Callable)
```

### 6.7 ASR 接口

```python
class BaseASR(ABC):
    """ASR 抽象接口"""
    @abstractmethod
    def transcribe(self, audio_data: np.ndarray, sample_rate: int = 16000) -> ASRResult
    
    @abstractmethod
    def transcribe_file(self, file_path: str) -> ASRResult
    
    @property
    @abstractmethod
    def is_available(self) -> bool

@dataclass
class ASRResult:
    text: str
    confidence: float = 0.0
    is_final: bool = True
    latency_ms: float = 0.0

class VoskASR(BaseASR):
    """本地 Vosk ASR 实现"""
    def __init__(self, model_path: str, sample_rate: int = 16000)

class CloudASR(BaseASR):
    """云端 ASR (百度/阿里云) 实现"""
    def __init__(self, provider: str, api_key: str, secret_key: str = None)
```

### 6.8 LLM 接口

```python
class BaseLLM(ABC):
    """LLM 抽象接口"""
    @abstractmethod
    def chat(self, 
             messages: List[Dict[str, str]], 
             stream: bool = False) -> Union[str, Iterator[str]]
    
    @abstractmethod
    def chat_with_context(self, 
                          user_text: str,
                          conversation_id: str = None) -> LLMResponse

@dataclass
class LLMResponse:
    text: str
    conversation_id: str
    tokens_used: int = 0
    latency_ms: float = 0.0

class DeepSeekLLM(BaseLLM):
    """DeepSeek API 客户端"""
    def __init__(self, api_key: str, model: str = "deepseek-chat", 
                 base_url: str = "https://api.deepseek.com",
                 system_prompt: str = None)
```

### 6.9 TTS 接口

```python
class BaseTTS(ABC):
    """TTS 抽象接口"""
    @abstractmethod
    def synthesize(self, text: str) -> TTSResult
    
    @abstractmethod
    def synthesize_stream(self, text: str) -> Iterator[bytes]
    
    @property
    @abstractmethod
    def is_available(self) -> bool

@dataclass
class TTSResult:
    audio_data: bytes
    format: str = "mp3"       # mp3 / wav / pcm
    sample_rate: int = 16000
    latency_ms: float = 0.0

class EdgeTTS(BaseTTS):
    """Microsoft Edge TTS 实现 (免费, 高质量)"""
    def __init__(self, voice: str = "zh-CN-XiaoxiaoNeural", 
                 rate: str = "+0%", pitch: str = "+0Hz")

class PiperTTS(BaseTTS):
    """Piper 本地 TTS 实现 (离线备份)"""
    def __init__(self, model_path: str, config_path: str = None)
```

### 6.10 Skill (技能接口)

```python
class BaseSkill(ABC):
    """技能抽象基类"""
    name: str
    description: str
    keywords: List[str]           # 触发关键词
    priority: int = 0            # 优先级 (越大越高)
    
    @abstractmethod
    def can_handle(self, text: str) -> bool
    
    @abstractmethod
    def execute(self, text: str, context: SkillContext) -> SkillResult
    
    @abstractmethod
    def get_response_text(self, result: SkillResult) -> str

@dataclass
class SkillContext:
    conversation_id: str
    user_id: str = "default"
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class SkillResult:
    success: bool
    data: Any = None
    response_text: str = ""
    error_message: str = ""

class SkillManager:
    """技能管理器: 注册、匹配、调度"""
    def register(self, skill: BaseSkill)
    def unregister(self, skill_name: str)
    def find_handler(self, text: str) -> Optional[BaseSkill]
    def execute(self, text: str, context: SkillContext) -> SkillResult
```

### 6.11 Engine (主引擎)

```python
class SmartSpeakerEngine:
    """
    主引擎: 创建、配置、启动所有模块
    协调模块间的数据流和生命周期
    """
    def __init__(self, config_path: str)
    def setup(self) -> None           # 初始化所有模块
    def start(self) -> None           # 启动
    def stop(self) -> None            # 优雅关闭
    def run_forever(self) -> None     # 阻塞运行
    def get_status(self) -> dict      # 获取运行状态
    
    # 内部生命周期:
    # setup() → _init_modules() → _wire_events() → _start_audio() → run_forever()
    #   → on_shutdown() → _stop_audio() → _cleanup()
```

---

## 7. 数据流和交互流程

### 7.1 一次完整交互的时序

```
时间轴 →

User:          [说出 "小智小智"]    [说出 "今天天气怎么样"]           [听回复]
              │                  │                           │
Mic:          ══════════════════════════════════════════════════════ (持续采集)
              │                  │                           │
WakeWord:     ─[检测到唤醒词]     │                           │
              │  publish(WAKE)   │                           │
State:        IDLE ──────────▶ LISTENING                     │
              │                  │                           │
VAD:          │                  ─[检测语音开始]              │
              │                  ─[累积音频缓冲]              │
              │                  ─[检测语音结束]              │
              │                  │  publish(SPEECH_END)      │
              │                  │  data: audio_buffer       │
State:        │                  LISTENING ─────────▶ THINKING
              │                  │                           │
ASR:         │                  ─ [转写音频]                 │
              │                  │  publish(ASR_RESULT)      │
              │                  │  data: "今天天气怎么样"     │
SkillMgr:    │                  ─ [匹配技能] → ChatSkill     │
              │                  │                           │
LLM:          │                  ─ [DeepSeek API 调用]       │
              │                  │  stream=True (流式)        │
              │                  │  publish(LLM_STREAM_CHUNK)│
              │                  │  publish(LLM_RESPONSE)    │
              │                  │                           │
TTS:          │                  ─ [将LLM响应转为语音]        │
              │                  │  publish(TTS_AUDIO_READY) │
              │                  │                           │
State:        │                  THINKING ──────────▶ SPEAKING
              │                  │                           │
Player:       │                  ─ [播放音频]                │
              │                  │  publish(PLAYBACK_START)  │
              │                  │                           ─ [播放中]
              │                  │  publish(PLAYBACK_DONE)   │
              │                  │                           │
State:        │                  SPEAKING ─────────▶ IDLE   │
              │                  │                           │
LED:          [绿色闪烁]         [蓝色/思考]                 [绿色常亮]
```

### 7.2 barge-in (打断) 流程

```
用户在播放时说出唤醒词:
SPEAKING state + WakeWord detected
  → Player.stop() (停止当前播放)
  → State → LISTENING (允许新的语音输入)
  → 新的交互循环开始
```

### 7.3 静音模式流程

```
Button pressed → publish(BUTTON_PRESSED)
  → State → MUTED
  → WakeWordDetector 暂停
  → LED 红色指示
Button pressed again → State → IDLE
  → WakeWordDetector 恢复
  → LED 绿色
```

---

## 8. 开机自启动方案

### 8.1 systemd 服务

使用 systemd 管理服务, 确保崩溃自动重启。

**文件: `/etc/systemd/system/smart-speaker.service`**

```ini
[Unit]
Description=Smart Speaker Voice Assistant
After=network-online.target sound.target pulseaudio.service
Wants=network-online.target sound.target
Requires=pulseaudio.service

[Service]
Type=simple
User=pi
Group=pi
WorkingDirectory=/home/pi/smart_speaker
Environment="PATH=/home/pi/smart_speaker_venv/bin:/usr/local/bin:/usr/bin:/bin"
Environment="PULSE_RUNTIME_PATH=/run/user/1000/pulse"
Environment="HOME=/home/pi"
ExecStartPre=/bin/sleep 10
ExecStart=/home/pi/smart_speaker_venv/bin/python src/main.py --config config/config.yaml
ExecStop=/bin/kill -SIGTERM $MAINPID
Restart=on-failure
RestartSec=5
StandardOutput=journal
StandardError=journal
SyslogIdentifier=smart-speaker

# 安全选项
NoNewPrivileges=yes
PrivateTmp=yes

[Install]
WantedBy=multi-user.target
```

### 8.2 启用服务

```bash
sudo cp scripts/service/smart-speaker.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable smart-speaker
sudo systemctl start smart-speaker

# 查看日志
journalctl -u smart-speaker -f
```

### 8.3 PulseAudio 用户态自动启动

```bash
# 确保 PulseAudio 以用户模式启动
systemctl --user enable pulseaudio
systemctl --user start pulseaudio

# 或通过桌面自动启动配置
mkdir -p ~/.config/systemd/user/
# 创建 pulseaudio.service 并启用
```

### 8.4 看门狗 (可选)

```bash
# 启用硬件看门狗 (防死锁)
sudo sed -i 's/#RuntimeWatchdogSec=0/RuntimeWatchdogSec=15/' /etc/systemd/system.conf
sudo sed -i 's/#RebootWatchdogSec=10min/RebootWatchdogSec=2min/' /etc/systemd/system.conf
```

---

## 9. 异常处理方案

### 9.1 异常分类

| 类别 | 场景 | 处理策略 |
|------|------|----------|
| **硬件异常** | USB麦克风断开 | 检测设备变化, 自动重连, 最多重试3次, 然后等待用户修复 |
| **硬件异常** | USB音响断开 | 同上, TTS 结果缓存到文件 |
| **网络异常** | WiFi断连 | 切换到离线模式 (Vosk+Piper), 网络恢复后切回 |
| **API异常** | DeepSeek 超时/限流 | 指数退避重试(1s→2s→4s→8s), 最多3次, 提示用户"网络不太好" |
| **API异常** | Edge TTS 不可用 | 自动切换 Piper 本地 TTS |
| **模型异常** | Vosk 模型加载失败 | 回退到云端 ASR, 记录日志 |
| **资源异常** | 内存不足 | 清理对话历史, 释放音频缓存, 必要时重启 |
| **状态异常** | 状态机卡死 | 设置超时定时器, 30s 无响应 → 强制回到 IDLE |
| **程序异常** | 未捕获异常 | systemd 自动重启 Restart=on-failure |

### 9.2 异常处理代码模式

```python
# 1. 模块级重试装饰器
def retry(max_retries: int = 3, backoff: float = 2.0, 
          exceptions: tuple = (Exception,)):
    """指数退避重试装饰器"""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    wait = backoff ** attempt
                    logger.warning(f"Retry {attempt+1}/{max_retries} for "
                                   f"{func.__name__}: {e}, waiting {wait}s")
                    time.sleep(wait)
            raise last_exception
        return wrapper
    return decorator

# 2. 优雅降级模式
class ASRManager:
    """ASR 管理器: 本地优先, 云端降级"""
    def __init__(self, local_asr, cloud_asr):
        self.primary = local_asr   # Vosk
        self.fallback = cloud_asr  # 云端 API
    
    def transcribe(self, audio):
        if self.primary.is_available:
            try:
                return self.primary.transcribe(audio)
            except ASRError:
                logger.warning("Local ASR failed, falling back to cloud")
        return self.fallback.transcribe(audio)

# 3. 超时保护
class TimeoutGuard:
    """超时保护: 状态机安全检查"""
    def __init__(self, timeout_ms: int = 30000):
        self.timeout_ms = timeout_ms
        self._timer = None
    
    def reset(self):
        if self._timer:
            self._timer.cancel()
        self._timer = threading.Timer(self.timeout_ms / 1000, self._on_timeout)
        self._timer.start()
    
    def _on_timeout(self):
        logger.error("State timeout! Forcing return to IDLE")
        EventBus().publish(Event.ERROR, source="timeout_guard", 
                          message="State timeout, forcing IDLE")
```

### 9.3 日志与告警

```python
# 分级日志
# - DEBUG: 开发调试
# - INFO: 正常交互记录 (唤醒、识别结果等)
# - WARNING: 降级、重试
# - ERROR: 模块失败、网络中断

# 日志轮转 (RotatingFileHandler)
# - 每个文件 10MB, 保留 5 个历史文件
# - 通过 journalctl 查看系统日志
```

---

## 10. 性能优化方案

### 10.1 延迟优化

| 优化点 | 方法 | 预期收益 |
|--------|------|----------|
| **唤醒词** | 使用 ONNX Runtime (非 PyTorch) | 更快启动 |
| **VAD** | Silero micro 模型, ONNX 推理 | <5ms per chunk |
| **ASR** | 流式识别 (Vosk partial results) | 感知延迟 -300ms |
| **LLM** | streaming=True (DeepSeek 流式) | 首字延迟 -2s |
| **TTS** | 预生成常用音频并缓存 | 延迟 -500ms |
| **TTS** | 流式合成, 边生成边播放 | 首音延迟 -1s |
| **音频** | 使用回调模式, 避免 buffer 累积 | 端到端 -200ms |

### 10.2 内存优化

| 优化点 | 方法 |
|--------|------|
| **模型加载** | 懒加载, 按需初始化 |
| **对话历史** | 限制最多 20 轮, 超出自动截断 |
| **音频缓存** | 使用 `tempfile` 存储临时文件, 定期清理 |
| **Python GC** | 及时 `del` 大对象, 必要时强制 `gc.collect()` |

### 10.3 CPU 优化

```python
# 1. 线程模型: 音频采集用独立高优先级线程
audio_thread = threading.Thread(target=audio_capture_loop, daemon=True)
audio_thread.priority = threading.HIGH_PRIORITY  # (需要 os.setpriority)

# 2. 音频帧批量处理: 减少 callback 调用频率
FRAMES_PER_BUFFER = 4096  # ~256ms @16kHz, 平衡延迟和CPU

# 3. ONNX Runtime 线程数设置
import onnxruntime as ort
ort_session = ort.InferenceSession("model.onnx", 
    providers=['CPUExecutionProvider'],
    sess_options=ort.SessionOptions())
ort_session.set_providers(['CPUExecutionProvider'], 
    [{'intra_op_num_threads': 2}])  # Pi3 用2线程, 留资源给其他模块
```

### 10.4 网络优化

```python
# 1. HTTP 连接复用
session = requests.Session()
adapter = requests.adapters.HTTPAdapter(
    pool_connections=5,
    pool_maxsize=10,
    max_retries=1
)
session.mount('https://', adapter)

# 2. DeepSeek API 流式调用
client.chat.completions.create(stream=True)  # 降低首字延迟

# 3. 音频预取: 在 LLM 返回时提前初始化 TTS 连接
```

### 10.5 离线/在线混合策略

```
┌──────────────────────────────────────────────┐
│                 网络检测                      │
│          ┌───────┴───────┐                   │
│          ▼               ▼                   │
│      在线模式         离线模式                │
│  ┌──────────────┐ ┌──────────────┐          │
│  │ ASR: Vosk本地 │ │ ASR: Vosk本地 │          │
│  │ LLM: DeepSeek │ │ LLM: 不可用   │          │
│  │ TTS: edge-tts │ │ TTS: Piper    │          │
│  └──────────────┘ └──────────────┘          │
│                                               │
│  注意: LLM 必须在线, 离线时提示用户           │
│  "网络不可用, 请检查网络连接"                 │
└──────────────────────────────────────────────┘
```

---

## 11. 后续扩展方案

### 11.1 智能家居控制 (Home Assistant)

```
┌─────────────┐    REST API    ┌──────────────────┐
│ Smart       │──────────────▶│ Home Assistant    │
│ Speaker     │               │ (同局域网)        │
│             │◀──────────────│                   │
│ Skills/HASS │    Webhook    │ 控制灯/空调/窗帘   │
└─────────────┘               └──────────────────┘

实现方式:
- hass_skill.py 通过 HASS REST API 或 WebSocket 通信
- 意图识别: "打开客厅灯" → intent: light.turn_on, entity: light.living_room
- 使用 DeepSeek Function Calling 提取意图
```

### 11.2 ROS/机器人控制

```
Smart Speaker ──(MQTT/HTTP)──▶ ROS Master ──▶ Robot Control Nodes

实现方式:
- 通过 MQTT bridge 或 rosbridge 通信
- 语音指令 → DeepSeek function calling → ROS Action/Topic
- "去厨房" → /move_base/goal (x, y, theta)
```

### 11.3 自定义技能开发

```python
# 技能热插拔: 放在 src/skills/ 目录自动发现
class MyCustomSkill(BaseSkill):
    name = "my_skill"
    keywords = ["打开电脑", "关机"]
    
    def can_handle(self, text: str) -> bool:
        return any(kw in text for kw in self.keywords)
    
    def execute(self, text: str, context: SkillContext) -> SkillResult:
        # 自定义逻辑
        subprocess.run(["wakeonlan", "00:11:22:33:44:55"])
        return SkillResult(success=True, response_text="已发送开机信号")
```

### 11.4 本地模型升级路径

当升级到 Raspberry Pi 5 (8GB) 时:

```
全本地方案:
├── 唤醒词: openWakeWord (本地)
├── VAD: Silero VAD (本地)
├── ASR: SenseVoice-Small ONNX (本地)  ← 升级
├── LLM: Ollama + Qwen2.5:1.5B (本地)  ← 升级
└── TTS: CosyVoice / Piper (本地)       ← 升级
```

### 11.5 多设备协同

```
┌────────┐     ┌────────┐     ┌────────┐
│ 客厅音箱│     │ 卧室音箱│     │ 厨房音箱│
└───┬────┘     └───┬────┘     └───┬────┘
    │              │              │
    └──────────────┼──────────────┘
                   │
            ┌──────▼──────┐
            │  MQTT Broker │  (Mosquitto)
            └──────┬──────┘
                   │
            ┌──────▼──────┐
            │  中心控制器  │  协调多设备状态
            └─────────────┘
```

### 11.6 功能扩展清单

- [ ] 闹钟 / 定时器
- [ ] 播客 / 音乐播放
- [ ] 蓝牙音箱模式 (A2DP sink)
- [ ] Web 管理后台
- [ ] OTA 更新
- [ ] 多语言支持
- [ ] 声纹识别 (用户区分)
- [ ] 情绪识别
- [ ] 屏幕扩展 (SPI LCD)

---

## 附录 A: 成本估算

| 项目 | 费用 |
|------|------|
| Raspberry Pi 3B+ | ~¥200 (已有) |
| USB 麦克风 | ~¥30-80 |
| USB 音响 | ~¥30-100 |
| DeepSeek API | ~¥1/百万tokens (个人用极少) |
| 其他云服务 (可选) | ~¥0-20/月 |
| **总计** | **¥260-400 一次性 + ¥0-20/月** |

## 附录 B: 参考资源

- openWakeWord: https://github.com/dscripka/openWakeWord
- Silero VAD: https://github.com/snakers4/silero-vad
- Vosk: https://alphacephei.com/vosk/
- DeepSeek API: https://platform.deepseek.com/
- edge-tts: https://github.com/rany2/edge-tts
- Piper TTS: https://github.com/rhasspy/piper
- FunASR: https://github.com/modelscope/FunASR
