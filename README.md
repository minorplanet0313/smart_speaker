# 🎙️ Smart Speaker — 智能语音助手

> 基于 Python 的类"小爱音箱"语音交互系统，支持 Raspberry Pi 和 x86_64 Linux

## 功能特性

- 🎤 **语音唤醒** — 说唤醒词唤醒设备 (基于 openWakeWord, ONNX 推理)
- 🗣️ **语音识别** — 本地离线 ASR (Vosk 中文模型) + 云端降级 (百度/阿里云)
- 🧠 **AI 对话** — 接入 DeepSeek 大语言模型，流式输出
- 🔊 **语音合成** — Edge TTS (免费高质量) + Piper 本地离线备份
- 💬 **多轮对话** — 上下文感知的连续对话，支持历史记忆
- 🔌 **可扩展** — 插件式技能框架，自定义功能只需一个文件
- 🛡️ **稳定运行** — systemd 守护, 状态超时保护, 崩溃自动恢复
- 📡 **优雅降级** — VAD 能量检测备份, ASR 云端备份, TTS 离线备份

## 硬件要求

| 硬件 | Raspberry Pi | x86_64 Linux 笔记本/PC |
|------|-------------|----------------------|
| CPU | Pi 3B+ / 4 / 5 | 任意 x86_64 (2核+) |
| RAM | ≥ 1GB | ≥ 4GB |
| 麦克风 | USB 麦克风 (推荐降噪) | 内置或 USB 麦克风 |
| 扬声器 | USB 音响 / 3.5mm 输出 | 内置或外接扬声器 |
| 网络 | WiFi 或以太网 | 需要 (ASR 离线可免) |
| GPIO | 可选 (LED + 按钮) | 不支持 |

> **已验证**: Ubuntu 20.04 / 22.04 x86_64 笔记本 (Intel SOF HDA DSP 内置声卡) 全部功能正常。

## 目录

- [快速开始 (5 分钟)](#快速开始-5-分钟)
- [详细安装指南](#详细安装指南)
  - [1. 克隆项目](#1-克隆项目)
  - [2. 安装系统依赖](#2-安装系统依赖)
  - [3. 配置 Python 环境](#3-配置-python-环境)
  - [4. 下载 AI 模型](#4-下载-ai-模型)
  - [5. 配置 API Key](#5-配置-api-key)
  - [6. 调整配置文件](#6-调整配置文件)
  - [7. 验证安装](#7-验证安装)
  - [8. 启动运行](#8-启动运行)
- [平台差异说明](#平台差异说明)
- [一键安装脚本 (Pi)](#一键安装脚本-pi)
- [配置文件说明](#配置文件说明)
- [工具命令](#工具命令)
- [唤醒词训练](#唤醒词训练)
- [自定义技能](#自定义技能)
- [故障排查](#故障排查)
- [项目结构](#项目结构)
- [License](#license)

---

## 快速开始 (5 分钟)

如果你已经配置好系统依赖和 Python 环境:

```bash
# 1. 激活虚拟环境
source ~/smart_speaker_venv/bin/activate
cd smart_speaker

# 2. 设置 API Key
cp .env.example .env
nano .env  # 填入 DEEPSEEK_API_KEY

# 3. (仅首次) 下载模型
mkdir -p models/openwakeword && python -c "from openwakeword.utils import download_models; download_models()"
wget https://alphacephei.com/vosk/models/vosk-model-small-cn-0.22.zip -O /tmp/vosk.zip
unzip -q /tmp/vosk.zip -d models/ && rm /tmp/vosk.zip

# 4. 启动
python src/main.py

# 5. 说 "Alexa" 唤醒 (或自行训练中文唤醒词后说 "小智小智")
```

---

## 详细安装指南

以下是从 `git clone` 到完整运行的每一步。

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
    ffmpeg \
    wget unzip

# 确保 PulseAudio 运行 (用户模式)
pulseaudio --check || pulseaudio --start
```

> **Python 版本要求**: ≥ 3.9。Ubuntu 20.04 默认 3.8, 需手动安装 3.9。Ubuntu 22.04+ 自带 3.10, 无需额外安装。

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

#### Fedora / CentOS

```bash
sudo dnf install -y python3.9 python3.9-devel python3-pip \
    portaudio-devel pulseaudio-libs-devel libsndfile alsa-lib-devel \
    ffmpeg wget unzip
```

#### macOS (开发/测试)

```bash
brew install python@3.9 portaudio pulseaudio libsndfile ffmpeg wget
```

### 3. 配置 Python 环境

```bash
# 3a. 创建虚拟环境 (Python 3.9+)
python3.9 -m venv ~/smart_speaker_venv

# 3b. 激活虚拟环境
source ~/smart_speaker_venv/bin/activate

# 3c. 升级 pip
pip install --upgrade pip setuptools wheel

# 3d. 如果你在 Raspberry Pi 上, 先注释掉不用的平台包
#     x86_64 上也需要跳过 gpiozero
sed -i 's/^gpiozero==/# gpiozero==/' requirements.txt

# 3e. 安装 Python 依赖
pip install -r requirements.txt

# 安装耗时约 5-15 分钟, 取决于网络和 CPU
# onnxruntime, torch (Silero VAD依赖) 等包含预编译 wheel, 
# PyAudio 需要从源码编译 (依赖 portaudio19-dev)
```

> **注意**: `silero-vad` 依赖 PyTorch (~900MB), 下载较慢。VAD 有能量检测降级, 即使安装失败也能正常工作。

<details>
<summary>可选: 仅安装核心依赖 (跳过 VAD / 唤醒词)</summary>

```bash
# 跳过 silero-vad (保留能量检测 VAD)
# 跳过 openwakeword (保留按键触发模式)
pip install pyaudio numpy PyYAML python-dotenv soundfile scipy requests httpx colorlog
pip install vosk onnxruntime openai edge-tts
```
</details>

### 4. 下载 AI 模型

#### 4a. Vosk ASR 模型 (语音识别, 必需)

```bash
mkdir -p models
cd models

# 中文小模型 (42MB, 适合 Pi, 精度足够日常使用)
wget https://alphacephei.com/vosk/models/vosk-model-small-cn-0.22.zip
unzip -q vosk-model-small-cn-0.22.zip
rm vosk-model-small-cn-0.22.zip

cd ..
```

> 其他 Vosk 模型: https://alphacephei.com/vosk/models  
> 中文大模型 `vosk-model-cn-0.22` (1.4GB) 精度更高但推理更慢。

#### 4b. openWakeWord 模型 (唤醒词, 可选)

```bash
mkdir -p models/openwakeword

# 下载预训练模型 (英文唤醒词 + 预处理)
python -c "
from openwakeword.utils import download_models
download_models(target_directory='models/openwakeword')
"
```

下载后得到 (约 15 个 .onnx / .tflite 文件):

| 文件名 | 唤醒词 |
|--------|--------|
| `alexa_v0.1.onnx` | "Alexa" |
| `hey_jarvis_v0.1.onnx` | "Hey Jarvis" |
| `hey_mycroft_v0.1.onnx` | "Hey Mycroft" |
| `timer_v0.1.onnx` | "Timer" |
| `weather_v0.1.onnx` | "Weather" |

> **注意**: openWakeWord 6.0+ 内置模型需额外下载。预处理模型 (`melspectrogram.onnx`, `embedding_model.onnx`) 也需放到包的 `resources/models/` 目录。安装脚本执行 `download_models()` 时会自动处理。

#### 4c. Piper TTS 离线模型 (可选)

```bash
mkdir -p models/piper-voices

# 下载中文语音模型
wget https://huggingface.co/rhasspy/piper-voices/resolve/main/zh/zh_CN/huayan/medium/zh_CN-huayan-medium.onnx \
    -O models/piper-voices/zh_CN-huayan-medium.onnx

wget https://huggingface.co/rhasspy/piper-voices/resolve/main/zh/zh_CN/huayan/medium/zh_CN-huayan-medium.onnx.json \
    -O models/piper-voices/zh_CN-huayan-medium.onnx.json
```

> 更多语音: https://huggingface.co/rhasspy/piper-voices

### 5. 配置 API Key

```bash
# 5a. 复制示例文件
cp .env.example .env

# 5b. 编辑 .env 填入真实 Key
nano .env
```

`.env` 文件内容:

```ini
# === 必需 ===
DEEPSEEK_API_KEY=sk-your-deepseek-api-key-here

# === 可选: 天气技能 ===
WEATHER_API_KEY=your-qweather-api-key

# === 可选: 云端 ASR 备份 ===
BAIDU_APP_ID=your-baidu-app-id
BAIDU_API_KEY=your-baidu-api-key
BAIDU_SECRET_KEY=your-baidu-secret-key
```

> **获取 API Key**:  
> - DeepSeek: https://platform.deepseek.com/ → 注册 → API Keys (有免费额度)  
> - 和风天气: https://dev.qweather.com/ → 注册 → 控制台 (免费 1000次/天)  
> - 百度 ASR: https://ai.baidu.com/tech/speech → 控制台 (免费 5万次/年)

### 6. 调整配置文件

编辑 `config/config.yaml`, 关键配置项:

```yaml
# === Raspberry Pi 配置 ===
wake_word:
  enabled: true
  model_path: "./models/openwakeword/alexa_v0.1.onnx"  # 暂用英文, 中文需训练
  inference_framework: "onnx"

gpio:
  enabled: true   # Pi 上启用 LED 和按钮

performance:
  onnx_threads: 2             # Pi 3/4 推荐 2, Pi 5 可设 4

# === x86_64 笔记本/PC 配置 ===
gpio:
  enabled: false              # 无 GPIO, 必须关闭!

performance:
  onnx_threads: 4             # x86_64 多核可设较高

# === 通用: 如无唤醒词模型, 可暂时禁用 ===
wake_word:
  enabled: false              # 关闭后需修改代码用按键触发
```

### 7. 验证安装

分步验证各模块是否正常:

```bash
# 激活环境
source ~/smart_speaker_venv/bin/activate

# 7a. 单元测试 (无需硬件, 1 秒内完成)
pytest tests/test_basic.py -v
# 预期: 12 passed

# 7b. 音频设备检测
python src/main.py --list-devices
# 预期: 列出麦克风和扬声器

# 7c. TTS 语音列表 (测试网络)
python src/main.py --list-voices
# 预期: 列出 14+ 个中文语音

# 7d. ASR 模型检测 (需要先下载 Vosk 模型)
python src/main.py --test-asr /dev/null 2>&1 | head -5
# 预期: 显示 "Vosk 模型加载完成" (然后报音频格式错——这是预期的)

# 7e. 安装完整性检查
python -c "
from src.audio.capture import AudioCapture
from src.audio.vad import VoiceActivityDetector
from src.wake_word.detector import WakeWordDetector
from src.asr.vosk_asr import VoskASR
from src.llm.deepseek import DeepSeekLLM
from src.tts.edge_tts import EdgeTTS
from src.skills.skill_manager import SkillManager
print('All modules OK')
"
```

### 8. 启动运行

```bash
# 8a. 确保代理设置正确 (如有 HTTP 代理而 sock 不被 httpx 支持)
unset ALL_PROXY all_proxy

# 8b. 设置 API Key (如未用 .env 文件)
export DEEPSEEK_API_KEY="sk-your-key"

# 8c. 启动
python src/main.py --log-level INFO

# 8d. 正常启动后输出:
#   ╔══════════════════════════════════════════╗
#   ║         🎙️  Smart Speaker               ║
#   ║         智能语音助手 v0.1.0               ║
#   ╚══════════════════════════════════════════╝
#   ✅ 初始化完成, 唤醒词: 小智小智
#   💡 说 '小智小智' 来唤醒我

# 8e. 开启 DEBUG 日志查看详细流程
python src/main.py --log-level DEBUG
```

#### systemd 开机自启动 (Raspberry Pi / Linux 服务器)

```bash
# 修改服务文件中的路径
sudo cp scripts/service/smart-speaker.service /etc/systemd/system/
sudo sed -i "s|/home/pi/smart_speaker|$(pwd)|g" /etc/systemd/system/smart-speaker.service
sudo sed -i "s|/home/pi/smart_speaker_venv|$HOME/smart_speaker_venv|g" /etc/systemd/system/smart-speaker.service
sudo sed -i "s|User=pi|User=$(whoami)|g" /etc/systemd/system/smart-speaker.service

# 启用并启动
sudo systemctl daemon-reload
sudo systemctl enable smart-speaker
sudo systemctl start smart-speaker

# 查看日志
journalctl -u smart-speaker -f
```

---

## 平台差异说明

| 特性 | Raspberry Pi | x86_64 Ubuntu 笔记本 |
|------|-------------|---------------------|
| Python | 3.9+ (系统自带) | 3.9+ (20.04 需手动安装) |
| GPIO | ✅ LED + 按钮 | ❌ 不支持 |
| 唤醒词 | 同左 | 同左 |
| ASR | 同左 | 同左 (推理更快) |
| TTS | 同左 | 同左 |
| PyAudio | 系统包安装 | 同左 (编译安装) |
| `onnxruntime` | `onnxruntime` (ARM wheel) | `onnxruntime` (x86_64 wheel, 自动选择) |
| `silero-vad` | 需要 torch (ARM wheel) | 需要 torch (x86_64 wheel, ~900MB) |
| 性能 | 建议 `onnx_threads: 2` | 建议 `onnx_threads: 4` |
| ALSA 警告 | 无 | 启动时有 ALSA `Unknown PCM` 警告 (无害) |

---

## 一键安装脚本 (Pi)

Raspberry Pi 用户可以使用一键安装脚本:

```bash
chmod +x scripts/install.sh
./scripts/install.sh
```

安装脚本会自动完成:
1. 系统依赖安装 (portaudio, pulseaudio, ffmpeg 等)
2. Swap 扩大至 2GB (编译 onnxruntime 需要)
3. PulseAudio 配置
4. Python 虚拟环境创建
5. pip 依赖安装
6. Vosk 模型下载
7. systemd 服务安装 (询问确认)

> x86_64 用户建议按照上方[详细安装指南](#详细安装指南)手动安装。

---

## 配置文件说明

完整配置文件: `config/config.yaml`

```yaml
# === 通用设置 ===
general:
  name: "小智音箱"            # 设备名称
  wake_word: "小智小智"       # 唤醒词文本 (仅显示用)
  language: "zh-CN"
  sample_rate: 16000
  log_level: "INFO"           # DEBUG | INFO | WARNING | ERROR

# === 音频设置 ===
audio:
  sample_rate: 16000
  channels: 1
  chunk_size: 1024            # 越小延迟越低, 越大 CPU 占用越少
  device:
    microphone: null          # null=系统默认, 或设备名
    speaker: null
  vad:
    enabled: true
    threshold: 0.4            # 语音检测灵敏度 [0, 1]
    min_speech_duration_ms: 250
    min_silence_duration_ms: 800
    max_speech_duration_ms: 15000

# === 唤醒词 ===
wake_word:
  enabled: true
  engine: "openwakeword"
  model_path: "./models/openwakeword/alexa_v0.1.onnx"
  threshold: 0.5              # 越高误触发越少, 但越难唤醒
  cooldown_ms: 2000

# === ASR ===
asr:
  primary: "vosk"             # 本地离线
  fallback: "cloud"           # 云端备份
  vosk:
    model_path: "./models/vosk-model-small-cn-0.22"

# === LLM ===
llm:
  provider: "deepseek"
  model: "deepseek-chat"
  api_key: "${DEEPSEEK_API_KEY}"   # 从 .env 或环境变量读取
  temperature: 0.7
  max_tokens: 1024

# === TTS ===
tts:
  primary: "edge"             # Edge TTS (在线, 高质量, 免费)
  fallback: "piper"           # Piper (离线)
  edge:
    voice: "zh-CN-XiaoxiaoNeural"

# === GPIO (仅 Pi) ===
gpio:
  enabled: false              # x86_64 必须 false
```

> 所有 `${VAR}` 语法的值从 `.env` 文件或环境变量中读取。

---

## 工具命令

```bash
# 列出所有音频输入/输出设备
python src/main.py --list-devices

# 列出 Edge TTS 所有可用中文语音
python src/main.py --list-voices

# 测试 Vosk ASR 识别 (指定音频文件)
python src/main.py --test-asr audio.wav

# 使用自定义配置文件
python src/main.py --config my_config.yaml

# 设置日志级别
python src/main.py --log-level DEBUG
```

---

## 唤醒词训练

openWakeWord 支持自定义唤醒词, 使用 Google Colab 免费训练:

### 步骤

1. **录制样本** (50-100 个):
   - 在不同距离/环境录制你说 "小智小智" 的音频
   - 每个样本 1-2 秒, 16kHz 单声道 WAV
   - 包含一些背景噪音样本 (电视声、风声等, 作为负样本)

2. **训练**:
   - 打开 openWakeWord Colab notebook: https://github.com/dscripka/openWakeWord
   - 上传你的音频样本
   - 运行训练流程 (~15 分钟)
   - 下载生成的 `.onnx` 文件

3. **部署**:
   ```bash
   cp xiao_zhi.onnx models/openwakeword/
   ```
   修改 `config/config.yaml`:
   ```yaml
   wake_word:
     model_path: "./models/openwakeword/xiao_zhi.onnx"
   ```

> **临时方案**: 在训练好中文唤醒词前, 项目已内置 openWakeWord 的 `alexa` 模型用于验证。你也可以禁用唤醒词 (`wake_word.enabled: false`), 改用按键触发模式。

---

## 自定义技能

在 `src/skills/builtin/` 创建新文件, 例如 `joke_skill.py`:

```python
import random
from src.skills.base import BaseSkill, SkillContext, SkillPriority, SkillResult

class JokeSkill(BaseSkill):
    name = "joke"
    keywords = ["笑话", "讲笑话", "来个段子"]
    priority = SkillPriority.NORMAL  # HIGH > NORMAL > LOW > FALLBACK

    def __init__(self):
        self._jokes = [
            "程序员最讨厌的字是什么？答：暂。",
            "为什么程序员总是分不清万圣节和圣诞节？因为 Oct 31 == Dec 25。",
        ]

    def can_handle(self, text: str) -> bool:
        return any(kw in text for kw in self.keywords)

    def execute(self, text: str, context: SkillContext) -> SkillResult:
        joke = random.choice(self._jokes)
        return SkillResult(
            success=True,
            response_text=joke,   # 直接返回文本, 不经 LLM
            needs_llm=False,      # 不需要 LLM 再处理
        )
```

然后注册到 `src/core/engine.py` 的 `_init_skills()`:

```python
def _init_skills(self) -> None:
    self.skill_manager = SkillManager()
    self.skill_manager.register(ChatSkill(self.llm))
    self.skill_manager.register(TimeSkill())
    self.skill_manager.register(JokeSkill())  # ← 添加这行
```

### 技能优先级规则

- `HIGH` — 精确匹配, 如时间/天气 (先于 LLM)
- `NORMAL` — 一般自定义技能
- `LOW` — 低优先级技能
- `FALLBACK` — ChatSkill 使用, 永远匹配, 路由到 LLM

技能可返回 `needs_llm=True` 将结果交由 LLM 二次处理; 返回 `response_text` 则直接合成语音输出。

---

## 故障排查

### 启动时 ALSA 大量 "Unknown PCM" 警告

**原因**: PyAudio 探测 ALSA 配置中不存在的设备。  
**影响**: 无。麦克风和扬声器使用 PulseAudio 通道, 不受影响。  
**解决**: 忽略, 或设置默认 ALSA 设备为 Pulse。

### silero-vad 推理失败: `'numpy.ndarray' object has no attribute 'dim'`

**原因**: silero-vad 5.x 要求 PyTorch tensor。  
**影响**: VAD 自动降级为能量检测 (仍可用)。  
**解决**: 已在最新代码中修复 (自动转换)。拉取最新代码即可。

### silero-vad 推理失败: `Provided number of samples is X (Supported values: 512 for 16000)`

**原因**: silero-vad 5.x 严格要求 512 帧输入。  
**影响**: 同上, 能量检测降级。  
**解决**: 已在 `src/audio/vad.py` 中添加帧缓冲逻辑。

### `ValueError: Unknown scheme for proxy URL URL('socks://...')`

**原因**: 系统设置了 `ALL_PROXY=socks://...`, httpx 不支持。  
**解决**: 启动前执行 `unset ALL_PROXY all_proxy`, 或在配置中设置 http 代理。

### `ModuleNotFoundError: No module named 'RPi.GPIO'`

**原因**: 在 x86_64 上运行, GPIO 库不可用。  
**影响**: 无。GPIO 已在配置中禁用, LED 模块自动降级为日志输出。  
**解决**: 无需处理。

### PyAudio 报错: `[Errno -9996] Invalid input device`

**原因**: 默认音频设备不可用。  
**解决**: 先用 `--list-devices` 查看设备列表, 然后在 `config.yaml` 中指定:
```yaml
audio:
  device:
    microphone: "pulse"  # 或具体设备名
```

### Edge TTS 下载失败 / 超时

**原因**: Edge TTS 使用微软 CDN, 国内可能较慢。  
**解决**: 配置代理或使用 Piper 离线 TTS:
```yaml
tts:
  primary: "piper"
```

---

## 项目结构

```
smart_speaker/
├── README.md
├── DESIGN.md                   # 完整技术方案文档
├── CLAUDE.md                   # Claude Code 项目指南
├── requirements.txt            # Python 依赖
├── .env.example                # 环境变量模板
├── config/
│   └── config.yaml             # 主配置文件 (单一配置源)
├── src/
│   ├── main.py                 # 入口, CLI 参数解析
│   ├── core/
│   │   ├── engine.py           # 主引擎: 初始化、事件连接、主循环
│   │   ├── event_bus.py        # 事件总线 (发布/订阅)
│   │   └── state_machine.py    # 状态机 (IDLE/LISTENING/THINKING/SPEAKING)
│   ├── audio/
│   │   ├── capture.py          # PyAudio 麦克风流式采集
│   │   ├── player.py           # 音频播放 (支持打断)
│   │   ├── vad.py              # Silero VAD + 能量检测降级
│   │   └── utils.py            # int16 ↔ float32 转换, RMS 计算
│   ├── wake_word/
│   │   └── detector.py         # openWakeWord 封装 (滑动窗口, 冷却)
│   ├── asr/
│   │   ├── base.py             # ASR 抽象基类 + ASRResult
│   │   ├── vosk_asr.py         # Vosk 离线 ASR
│   │   └── cloud_asr.py        # 云端 ASR (百度/阿里云)
│   ├── llm/
│   │   ├── base.py             # LLM 抽象基类
│   │   ├── deepseek.py         # DeepSeek (OpenAI SDK, 流式, 重试)
│   │   └── context.py          # 对话上下文管理 (历史, 超时)
│   ├── tts/
│   │   ├── base.py             # TTS 抽象基类 + TTSResult
│   │   ├── edge_tts.py         # Edge TTS (Microsoft, 免费在线)
│   │   └── piper_tts.py        # Piper TTS (本地离线)
│   ├── skills/
│   │   ├── base.py             # 技能抽象基类 + SkillContext + SkillResult
│   │   ├── skill_manager.py    # 技能管理器 (优先级排序)
│   │   └── builtin/
│   │       ├── chat_skill.py   # 通用聊天 (FALLBACK, 路由到 LLM)
│   │       ├── time_skill.py   # 时间/日期/星期查询
│   │       └── weather_skill.py # 天气查询 (和风 API)
│   ├── gpio/
│   │   ├── led.py              # LED 状态指示灯 (单色/RGB, 降级为日志)
│   │   └── button.py           # 物理按钮 (静音/快捷操作)
│   └── utils/
│       ├── config.py           # 配置加载 (YAML + ${ENV_VAR} + .env)
│       └── logger.py           # 日志 (控制台颜色 + 文件轮转)
├── scripts/
│   ├── install.sh              # Pi 一键安装脚本
│   ├── start.sh                # 启动脚本
│   └── service/
│       └── smart-speaker.service  # systemd 服务模板
├── models/
│   ├── vosk-model-small-cn-0.22/  # Vosk ASR 模型 (下载后)
│   ├── openwakeword/              # 唤醒词模型 (下载后)
│   └── piper-voices/             # Piper TTS 模型 (下载后, 可选)
└── tests/
    └── test_basic.py           # 基础功能测试 (无需硬件)
```

## 架构速览

```
USB/内置麦克风 → AudioCapture (PyAudio, 独立线程)
                      ↓
              WakeWordDetector (IDLE状态, openWakeWord ONNX)
                      ↓ 唤醒!
              VAD (Silero ONNX, LISTENING状态, 能量降级)
                      ↓ 语音结束
              ASR (Vosk本地离线 → 云端百度/阿里降级, 独立线程)
                      ↓ 文本
              SkillManager (优先级匹配: TimeSkill > WeatherSkill > ChatSkill)
                      ↓
              LLM (DeepSeek, OpenAI SDK流式输出)
                      ↓
              TTS (Edge TTS在线 → Piper离线降级, 独立线程)
                      ↓
              AudioPlayer → 内置/USB扬声器
```

## License

MIT
