# 🎙️ Smart Speaker — 智能语音助手

> 基于 Raspberry Pi 3B+ 的类"小爱音箱"语音交互系统

## 功能特性

- 🎤 **语音唤醒** — 说"小智小智"唤醒设备 (基于 openWakeWord)
- 🗣️ **语音识别** — 本地离线 ASR (Vosk) + 云端降级
- 🧠 **AI 对话** — 接入 DeepSeek 大语言模型
- 🔊 **语音合成** — Edge TTS (免费高质) + Piper 本地备份
- 💬 **多轮对话** — 上下文感知的连续对话
- 🔌 **可扩展** — 技能框架, 支持自定义功能
- 🛡️ **稳定运行** — systemd 守护, 崩溃自动重启

## 硬件要求

| 硬件 | 说明 |
|------|------|
| Raspberry Pi 3B+ | 或 Pi 4/5 (性能更好) |
| USB 麦克风 | 推荐降噪麦克风 |
| USB 音响 | 或 3.5mm 音频输出 |
| 网络连接 | WiFi 或以太网 |

## 快速开始

### 1. 克隆项目

```bash
git clone <your-repo-url> smart_speaker
cd smart_speaker
```

### 2. 一键安装

```bash
chmod +x scripts/install.sh
./scripts/install.sh
```

### 3. 配置 API Key

```bash
cp .env.example .env
nano .env  # 填入 DEEPSEEK_API_KEY
```

### 4. 启动

```bash
source ~/smart_speaker_venv/bin/activate
python src/main.py
```

### 5. (可选) 开机自启动

安装脚本会询问是否安装 systemd 服务, 或手动安装:

```bash
sudo cp scripts/service/smart-speaker.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable smart-speaker
sudo systemctl start smart-speaker

# 查看日志
journalctl -u smart-speaker -f
```

## 项目结构

```
smart_speaker/
├── DESIGN.md              # 完整技术方案文档
├── README.md
├── config/
│   └── config.yaml        # 主配置文件
├── src/
│   ├── main.py            # 入口
│   ├── core/              # 引擎、状态机、事件总线
│   ├── audio/             # 音频捕获、播放、VAD
│   ├── wake_word/         # 唤醒词检测
│   ├── asr/               # 语音识别 (Vosk/云端)
│   ├── llm/               # 大模型 (DeepSeek)
│   ├── tts/               # 语音合成 (Edge/Piper)
│   ├── skills/            # 技能框架
│   ├── gpio/              # GPIO 控制
│   └── utils/             # 工具
├── scripts/
│   ├── install.sh         # 安装脚本
│   ├── start.sh           # 启动脚本
│   └── service/           # systemd 服务
└── tests/
```

## 唤醒词训练

使用 openWakeWord 提供的 Colab 笔记本训练自定义唤醒词:

1. 录制 50-100 个"小智小智"语音样本
2. 在 Colab 中训练: https://github.com/dscripka/openWakeWord
3. 导出 .onnx 模型放到 `models/openwakeword/`
4. 更新 `config/config.yaml` 中的 `wake_word.model_path`

## 工具命令

```bash
# 列出音频设备
python src/main.py --list-devices

# 列出 Edge TTS 可用语音
python src/main.py --list-voices

# 测试 ASR 识别
python src/main.py --test-asr audio.wav
```

## 自定义技能

在 `src/skills/builtin/` 目录创建新技能:

```python
from src.skills.base import BaseSkill, SkillContext, SkillPriority, SkillResult

class MySkill(BaseSkill):
    name = "my_skill"
    keywords = ["打开电脑"]
    priority = SkillPriority.NORMAL

    def can_handle(self, text: str) -> bool:
        return any(kw in text for kw in self.keywords)

    def execute(self, text: str, context: SkillContext) -> SkillResult:
        # 你的业务逻辑
        return SkillResult(success=True, response_text="已执行")
```

然后在 `src/core/engine.py` 的 `_init_skills()` 方法中注册。

## 完整文档

详细技术方案请阅读 [DESIGN.md](DESIGN.md)

## License

MIT
