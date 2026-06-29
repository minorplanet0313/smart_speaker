# 🎙️ Smart Speaker — 智能语音助手

> 基于 Python 的类"小爱音箱"语音交互系统，支持 Raspberry Pi 3B+/4/5 和 x86_64 Linux

## 功能特性

- 🎤 **语音唤醒** — 说唤醒词唤醒设备 (openWakeWord, ONNX 推理, 支持自定义中文唤醒词)
- 🗣️ **多引擎语音识别** — 本地 sherpa-onnx (14MB, ~93%) / Vosk (42MB, ~85%) + 云端降级 (百度/腾讯)
- 🧠 **流式 AI 对话** — DeepSeek LLM 流式输出 + 分句 TTS，首响延迟 < 1 秒
- 🔊 **语音合成** — Edge TTS (免费高质量) / Piper 本地离线，主备自动切换
- 🎛️ **Web 管理面板** — 实时仪表盘 (状态/对话/事件流/配置编辑/远程控制)
- 💬 **多轮对话** — 上下文感知的连续对话，支持历史记忆
- 🔌 **可扩展** — 插件式技能框架，自定义功能只需一个文件
- 🛡️ **优雅降级** — VAD 能量检测备份, ASR 云端备份, TTS 离线备份, 三级降级链路

---

## 硬件要求

| 硬件 | Raspberry Pi | x86_64 Linux 笔记本/PC |
|------|-------------|----------------------|
| CPU | Pi 3B+ / 4 / 5 | 任意 x86_64 (2核+) |
| RAM | ≥ 1GB | ≥ 4GB |
| 麦克风 | USB 麦克风 | 内置或 USB 麦克风 |
| 扬声器 | USB 音响 / 3.5mm | 内置或外接扬声器 |
| 网络 | WiFi 或以太网 | 需要 |
| GPIO | 可选 (LED + 按钮) | 不支持 |

> **已验证**: Ubuntu 20.04 / 22.04 x86_64, Raspberry Pi 3B+ 全部功能正常。

---

## 快速开始 (5 分钟)

```bash
# 1. 激活虚拟环境
source ~/smart_speaker_venv/bin/activate
cd smart_speaker

# 2. 设置 API Key
cp .env.example .env
nano .env  # 填入 DEEPSEEK_API_KEY

# 3. (首次) 下载模型
mkdir -p models/openwakeword
python -c "from openwakeword.utils import download_models; download_models(target_directory='models/openwakeword')"

# Vosk 小模型 (42MB, Pi 3B+ 推荐)
wget https://alphacephei.com/vosk/models/vosk-model-small-cn-0.22.zip -O /tmp/vosk.zip
unzip -q /tmp/vosk.zip -d models/ && rm /tmp/vosk.zip

# 4. 启动
python src/main.py

# 5. 说唤醒词唤醒，或按回车 (键盘唤醒模式)
```

---

## 目录

- [快速开始](#快速开始-5-分钟)
- [详细安装指南](#详细安装指南)
- [平台差异说明](#平台差异说明)
- [一键安装脚本 (Pi)](#一键安装脚本-pi)
- [配置文件说明](#配置文件说明)
- [Web 管理面板](#web-管理面板)
- [ASR 引擎选择](#asr-引擎选择)
- [工具命令](#工具命令)
- [唤醒词训练](#唤醒词训练)
- [自定义技能](#自定义技能)
- [故障排查](#故障排查)
- [架构速览](#架构速览)
- [项目结构](#项目结构)

---

## 详细安装指南

### 1. 克隆项目

```bash
git clone <your-repo-url> smart_speaker
cd smart_speaker
```

### 2. 安装系统依赖

#### Ubuntu / Debian (20.04+)

```bash
sudo apt update
sudo apt install -y \
    python3.9 python3.9-venv python3.9-dev \
    python3-pip \
    portaudio19-dev libportaudio2 \
    pulseaudio pulseaudio-utils \
    libsndfile1 libasound2-dev \
    ffmpeg wget unzip

pulseaudio --check || pulseaudio --start
```

> Python ≥ 3.9。Ubuntu 20.04 默认 3.8, 需手动安装 3.9。22.04+ 自带 3.10。

#### Raspberry Pi (Raspberry Pi OS Bookworm)

```bash
sudo apt update
sudo apt install -y \
    python3 python3-pip python3-venv python3-dev \
    portaudio19-dev python3-pyaudio \
    pulseaudio pulseaudio-utils \
    libsndfile1 libasound2-dev \
    libatlas-base-dev \
    git wget curl ffmpeg
```

### 3. 配置 Python 环境

```bash
python3.9 -m venv ~/smart_speaker_venv
source ~/smart_speaker_venv/bin/activate
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

> **注意**: `silero-vad` 依赖 PyTorch (~900MB)。VAD 有能量检测降级，即使安装失败也能正常工作。

### 4. 下载 AI 模型

#### 4a. ASR 模型 (语音识别, 必需)

```bash
mkdir -p models
cd models

# 推荐: Vosk 中文小模型 (42MB, 适合 Pi 3B+)
wget https://alphacephei.com/vosk/models/vosk-model-small-cn-0.22.zip
unzip -q vosk-model-small-cn-0.22.zip
rm vosk-model-small-cn-0.22.zip

# 可选: sherpa-onnx 模型 (14MB, 更准更快)
# wget https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-streaming-zipformer-zh-14M-2025-06-30.tar.bz2
# tar xvf sherpa-onnx-streaming-zipformer-zh-14M-2025-06-30.tar.bz2 && rm *.tar.bz2

# 可选: Vosk 大模型 (1.3GB, Pi 5 适用)
# wget https://alphacephei.com/vosk/models/vosk-model-cn-0.22.zip

cd ..
```

#### 4b. openWakeWord 模型 (唤醒词)

```bash
mkdir -p models/openwakeword
python -c "
from openwakeword.utils import download_models
download_models(target_directory='models/openwakeword')
"
```

内置模型包括 `alexa`, `hey_jarvis`, `hey_mycroft` 等英文唤醒词。中文唤醒词模型 `xiao_zhi.onnx` 已包含在项目中。

#### 4c. Piper TTS 离线模型 (可选)

```bash
mkdir -p models/piper-voices
wget https://huggingface.co/rhasspy/piper-voices/resolve/main/zh/zh_CN/huayan/medium/zh_CN-huayan-medium.onnx \
    -O models/piper-voices/zh_CN-huayan-medium.onnx
wget https://huggingface.co/rhasspy/piper-voices/resolve/main/zh/zh_CN/huayan/medium/zh_CN-huayan-medium.onnx.json \
    -O models/piper-voices/zh_CN-huayan-medium.onnx.json
```

### 5. 配置 API Key

```bash
cp .env.example .env
nano .env
```

```ini
# 必需
DEEPSEEK_API_KEY=sk-your-deepseek-api-key

# 可选: 天气技能
WEATHER_API_KEY=your-qweather-api-key

# 可选: 云端 ASR 备份
BAIDU_API_KEY=your-baidu-api-key
BAIDU_SECRET_KEY=your-baidu-secret-key
TENCENT_SECRET_ID=your-tencent-secret-id
TENCENT_SECRET_KEY=your-tencent-secret-key
```

> DeepSeek: https://platform.deepseek.com/ | 和风天气: https://dev.qweather.com/ | 百度 ASR: https://ai.baidu.com/tech/speech (5万次免费) | 腾讯 ASR: https://cloud.tencent.com/product/asr (5千次/月免费)

### 6. 验证安装

```bash
source ~/smart_speaker_venv/bin/activate

# 单元测试 (无需硬件)
pytest tests/test_basic.py -v
# 预期: 18 passed

# 音频设备检测
python src/main.py --list-devices

# TTS 语音列表
python src/main.py --list-voices

# ASR 模型检测
python -c "from src.asr.vosk_asr import VoskASR; a=VoskASR(); print('OK:', a.is_available)"
```

### 7. 启动运行

```bash
python src/main.py --log-level INFO

# 调试模式
python src/main.py --log-level DEBUG

# 键盘唤醒模式 (无需说唤醒词, 按回车触发)
# 启动后按回车即可对话
```

#### systemd 开机自启动

```bash
sudo cp scripts/service/smart-speaker.service /etc/systemd/system/
sudo sed -i "s|/home/pi/smart_speaker|$(pwd)|g" /etc/systemd/system/smart-speaker.service
sudo sed -i "s|User=pi|User=$(whoami)|g" /etc/systemd/system/smart-speaker.service
sudo systemctl daemon-reload
sudo systemctl enable --now smart-speaker
journalctl -u smart-speaker -f
```

---

## 平台差异说明

| 特性 | Raspberry Pi | x86_64 Ubuntu |
|------|-------------|---------------|
| GPIO | ✅ LED + 按钮 | ❌ |
| 推荐 ASR | sherpa-onnx (14MB) 或 Vosk 小模型 (42MB) | 任意模型 |
| onnx_threads | 2 | 4 |
| 内存压力 | 较高 (1GB 共 4-5 个模型) | 充足 |

---

## 一键安装脚本 (Pi)

```bash
chmod +x scripts/install.sh
./scripts/install.sh
```

自动完成: 系统依赖 → Swap → Python 环境 → pip → 模型下载 → systemd 服务。

---

## 配置文件说明

完整配置: `config/config.yaml`。支持 `${ENV_VAR}` 语法从 `.env` 读取敏感值。

### 关键配置项

```yaml
# === ASR 引擎 ===
asr:
  primary: "vosk"          # vosk | sherpa (本地) | cloud
  fallback: "cloud"        # 本地失败 → 云端自动降级
  incremental: true        # 边听边识别 (降低感知延迟 ~500ms)
  preprocess: true         # 音频预处理 (去噪+归一化, 提升准确率)
  vosk:
    model_path: "./models/vosk-model-small-cn-0.22"
  sherpa:
    model_path: "./models/sherpa-onnx-streaming-zipformer-zh-14M"
    num_threads: 2
  cloud:
    provider: "baidu"      # baidu | tencent | aliyun

# === LLM ===
llm:
  stream: true             # 流式输出 → 分句 TTS (首响延迟 < 1s)
  retry:
    max_retries: 3
    backoff_base: 2

# === TTS ===
tts:
  primary: "edge"          # edge | piper
  fallback: "piper"        # 主失败 → 自动切换

# === VAD ===
audio:
  vad:
    min_silence_duration_ms: 400   # 语音结束判定 (越小响应越快)
    min_speech_duration_ms: 100

# === 调试 ===
debug:
  save_audio: false        # 保存每次录音用于调试 (默认关)

# === Web 管理面板 ===
web:
  enabled: false           # 启用后访问 http://<pi-ip>:8080
  port: 8080
```

---

## Web 管理面板

启用后浏览器访问 `http://<pi-ip>:8080`:

```
┌─────────────────────────────────────────┐
│  🟢 小智音箱    idle    唤醒    关闭Web  │
├─────────────────────────────────────────┤
│ 系统状态          │ 音频电平             │
│ 状态: idle       │ RMS: 0.0012         │
│ 交互: 42 次      │ ██░░░░░░░           │
│ 错误: 0          │ 采集 ✓  播放 空闲    │
├─────────────────────────────────────────┤
│ 系统资源                                │
│ CPU 12%  ████░░░░░░                     │
│ 内存 345 / 976 MB                       │
├─────────────────────────────────────────┤
│ [对话] [事件流] [配置]                   │
│                                         │
│ 用户: 现在几点                          │
│ 小智: 现在是下午3点15分                  │
│                                         │
│ [输入文字指令...              ] [发送]   │
└─────────────────────────────────────────┘
```

功能: 实时状态 · 对话记录 · 事件流 · 配置编辑 · 远程唤醒 · 文字输入

```bash
pip install bottle psutil
# config.yaml: web.enabled: true → 重启
```

---

## ASR 引擎选择

| 引擎 | 类型 | 模型大小 | 准确率 | 延迟 | 费用 |
|------|------|---------|--------|------|------|
| **sherpa-onnx** | 本地 | 14MB | ~93% | <300ms | 免费 |
| **Vosk (小)** | 本地 | 42MB | ~85% | 150-400ms | 免费 |
| **Vosk (大)** | 本地 | 1.3GB | ~92% | 300-800ms | 免费 |
| **百度** | 云端 | - | ~95% | 200-500ms | 5万次终身免费 |
| **腾讯** | 云端 | - | ~93% | 200-600ms | 5千次/月免费 |

Pi 3B+ 推荐 `sherpa-onnx` (14MB, 最轻) 或 `Vosk 小模型` (42MB)。大模型 (1.3GB) 在 1GB 内存上会触发 swap。

降级链路: **本地 (sherpa/vosk) → 云端 (baidu/tencent) → ERROR**

---

## 工具命令

```bash
# 音频设备列表
python src/main.py --list-devices

# Edge TTS 语音列表
python src/main.py --list-voices

# 测试 ASR (对比原始 vs 预处理后)
python src/main.py --test-asr audio.wav

# 自定义配置
python src/main.py --config my_config.yaml

# 调试日志
python src/main.py --log-level DEBUG

# 唤醒词诊断
python scripts/diagnose_wakeup.py

# 实时唤醒词测试 (观察置信度)
python scripts/test_wake_live.py --model models/openwakeword/xiao_zhi.onnx --threshold 0.5
```

---

## 唤醒词训练

openWakeWord 支持自定义唤醒词:

1. **录制样本** (50-100 个): 在不同距离/环境录制, 1-2s, 16kHz WAV
2. **训练**: 使用 [openWakeWord Colab](https://github.com/dscripka/openWakeWord) (~15 分钟)
3. **部署**: 复制 `.onnx` 到 `models/openwakeword/`, 修改 `config.yaml`

项目已内置中文唤醒词 `xiao_zhi.onnx` ("小智小智")。

---

## 自定义技能

在 `src/skills/builtin/` 创建新文件:

```python
import random
from src.skills.base import BaseSkill, SkillContext, SkillPriority, SkillResult

class JokeSkill(BaseSkill):
    name = "joke"
    keywords = ["笑话", "讲笑话", "来个段子"]
    priority = SkillPriority.NORMAL

    def __init__(self):
        self._jokes = ["程序员最讨厌的字是什么？答：暂。", "..."]

    def can_handle(self, text: str) -> bool:
        return any(kw in text for kw in self.keywords)

    def execute(self, text: str, context: SkillContext) -> SkillResult:
        return SkillResult(success=True, response_text=random.choice(self._jokes))
```

在 `engine.py` 的 `_init_skills()` 注册即可。技能返回 `response_text` 直接播报, 返回 `needs_llm=True` 路由到 LLM。

---

## 故障排查

### 启动时 ALSA "Unknown PCM" 警告

无害。麦克风和扬声器使用 PulseAudio 通道。

### PyAudio 报错: `Invalid input device`

`python src/main.py --list-devices` 查看设备, 在 `config.yaml` 中指定:
```yaml
audio:
  device:
    microphone: "pulse"
```

### ModuleNotFoundError: No module named 'RPi.GPIO'

x86_64 上 GPIO 不可用。已在配置中禁用, 不影响运行。

### Edge TTS 下载失败 / 超时

切换离线 Piper:
```yaml
tts:
  primary: "piper"
```

### Vosk 模型未找到

检查路径: `ls models/vosk-model-small-cn-0.22/`。确认 `asr.vosk.model_path` 配置正确。

---

## 架构速览

```
USB/内置麦克风
  → AudioCapture (PyAudio, 独立线程)
    → Audio Queue → Audio Worker (解耦采集与处理)
      → WakeWordDetector (IDLE, openWakeWord ONNX)
        → VAD (Silero ONNX, 能量降级, LISTENING)
          → 增量 ASR feed (边听边识别)
            → 音频预处理 (去直流+高通+归一化)
              → ASR (sherpa-onnx/Vosk 本地 → 百度/腾讯云端降级)
                → SkillManager (Time/Weather/Chat)
                  → LLM (DeepSeek 流式 → 分句 TTS)
                    → TTS (Edge/Piper)
                      → AudioPlayer → 扬声器

Web UI ← EventBus ← Bottle HTTP + SSE (可选, :8080)
```

### 降级链路

```
唤醒词: openWakeWord → (关闭时键盘唤醒)
VAD:     Silero ONNX → 能量检测
ASR:     sherpa/Vosk → 百度云 → 腾讯云
TTS:     Edge TTS → Piper 离线
```

---

## 项目结构

```
smart_speaker/
├── README.md
├── DESIGN.md
├── CLAUDE.md
├── requirements.txt
├── .env.example
├── config/
│   └── config.yaml
├── src/
│   ├── main.py
│   ├── core/
│   │   ├── engine.py          # 主引擎 (queue offload + 流式管线)
│   │   ├── event_bus.py       # 事件总线 (pub/sub)
│   │   └── state_machine.py   # 状态机
│   ├── audio/
│   │   ├── capture.py         # PyAudio 采集
│   │   ├── player.py          # 播放 + 流式播放
│   │   ├── vad.py             # Silero VAD + 能量检测
│   │   ├── preprocessing.py   # 音频预处理 (去噪/归一化)
│   │   ├── ring_buffer.py     # 环形缓冲区
│   │   └── utils.py
│   ├── wake_word/
│   │   └── detector.py        # openWakeWord (lazy load)
│   ├── asr/
│   │   ├── base.py
│   │   ├── vosk_asr.py        # Vosk 离线 (增量接口)
│   │   ├── sherpa_asr.py      # sherpa-onnx (增量接口, 14MB)
│   │   └── cloud_asr.py       # 百度/腾讯/阿里云
│   ├── llm/
│   │   ├── deepseek.py        # DeepSeek (流式, 重试)
│   │   └── context.py         # 对话上下文
│   ├── tts/
│   │   ├── edge_tts.py        # Edge TTS (在线)
│   │   └── piper_tts.py       # Piper (离线, 模型缓存)
│   ├── skills/
│   │   ├── base.py
│   │   ├── skill_manager.py
│   │   └── builtin/
│   │       ├── chat_skill.py
│   │       ├── time_skill.py
│   │       └── weather_skill.py
│   ├── web/                   # Web 管理面板 (可选)
│   │   ├── collector.py       # 事件收集器
│   │   ├── server.py          # Bottle HTTP + SSE
│   │   └── static/index.html  # Alpine.js 仪表盘
│   ├── gpio/                  # (Pi 可选)
│   └── utils/
│       ├── config.py
│       ├── logger.py
│       ├── messages.py        # 统一错误话术
│       └── sentence_split.py  # 流式分句
├── scripts/
│   ├── install.sh
│   ├── diagnose_wakeup.py
│   ├── test_wake_live.py
│   └── service/
├── models/                    # 下载后
└── tests/
    └── test_basic.py          # 18 个测试
```

---

## License

MIT
