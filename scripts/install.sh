#!/usr/bin/env bash
# ============================================================
# Smart Speaker — 一键安装脚本
# 适用: Raspberry Pi 3B+ / Raspberry Pi OS Lite (Bookworm)
# ============================================================
#
# 用法:
#   chmod +x scripts/install.sh
#   ./scripts/install.sh
#
# 安装内容:
#   1. 系统依赖 (portaudio, pulseaudio, etc.)
#   2. Python 虚拟环境
#   3. Python 包依赖
#   4. 下载 AI 模型 (Vosk, 唤醒词)
#   5. 配置 systemd 服务 (可选)
#   6. 扩大 swap (编译需要)
# ============================================================

set -e

# ---- 颜色输出 ----
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

info()  { echo -e "${GREEN}[INFO]${NC}  $1"; }
warn()  { echo -e "${YELLOW}[WARN]${NC}  $1"; }
error() { echo -e "${RED}[ERROR]${NC} $1"; }
step()  { echo -e "\n${BLUE}========================================${NC}"; echo -e "${BLUE}$1${NC}"; echo -e "${BLUE}========================================${NC}"; }

# ---- 检查是否为 Raspberry Pi ----
if [ -f /proc/device-tree/model ]; then
    PI_MODEL=$(tr -d '\0' < /proc/device-tree/model)
    info "检测到: $PI_MODEL"
else
    warn "未检测到 Raspberry Pi, 继续安装 (用于开发/测试)"
fi

# ---- 项目根目录 ----
PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_DIR"
info "项目目录: $PROJECT_DIR"

# ============================================================
# Step 1: 系统更新和依赖安装
# ============================================================
step "Step 1/7: 安装系统依赖"

sudo apt update

sudo apt install -y \
    python3 python3-pip python3-venv python3-dev \
    portaudio19-dev python3-pyaudio \
    pulseaudio pulseaudio-utils \
    libsndfile1 libasound2-dev \
    libatlas-base-dev \
    git wget curl \
    libportaudio2 \
    ffmpeg

info "系统依赖安装完成"

# ============================================================
# Step 2: 扩大 Swap (Raspberry Pi 需要)
# ============================================================
step "Step 2/7: 配置 Swap (2GB)"

if [ -f /etc/dphys-swapfile ]; then
    CURRENT_SWAP=$(grep CONF_SWAPSIZE /etc/dphys-swapfile | cut -d= -f2)
    info "当前 Swap 大小: ${CURRENT_SWAP}MB"

    if [ "$CURRENT_SWAP" -lt 2048 ]; then
        info "扩大到 2048MB..."
        sudo dphys-swapfile swapoff
        sudo sed -i 's/^CONF_SWAPSIZE=.*/CONF_SWAPSIZE=2048/' /etc/dphys-swapfile
        sudo dphys-swapfile setup
        sudo dphys-swapfile swapon
        info "Swap 已扩大到 2048MB"
    else
        info "Swap 大小已足够"
    fi
else
    warn "未找到 dphys-swapfile, 跳过 Swap 配置"
fi

# ============================================================
# Step 3: 配置 PulseAudio
# ============================================================
step "Step 3/7: 配置 PulseAudio"

# 确保 PulseAudio 以用户模式运行
if ! pulseaudio --check 2>/dev/null; then
    info "启动 PulseAudio (用户模式)..."
    pulseaudio --start --daemonize=false --exit-idle-time=-1 &
    sleep 2
fi

# 显示当前音频设备
info "当前音频设备:"
echo "--- 输入 (麦克风) ---"
pactl list short sources 2>/dev/null | head -5 || echo "  无法获取"
echo "--- 输出 (扬声器) ---"
pactl list short sinks 2>/dev/null | head -5 || echo "  无法获取"

# ============================================================
# Step 4: 创建 Python 虚拟环境
# ============================================================
step "Step 4/7: 创建 Python 虚拟环境"

VENV_DIR="$HOME/smart_speaker_venv"

if [ -d "$VENV_DIR" ]; then
    info "虚拟环境已存在: $VENV_DIR"
else
    python3 -m venv "$VENV_DIR"
    info "虚拟环境已创建: $VENV_DIR"
fi

source "$VENV_DIR/bin/activate"

# 升级 pip
pip install --upgrade pip setuptools wheel

# 使用清华镜像加速 (国内)
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple 2>/dev/null || true

# ============================================================
# Step 5: 安装 Python 依赖
# ============================================================
step "Step 5/7: 安装 Python 依赖"

info "安装 Python 包 (这可能需要几分钟, 特别是编译 onnxruntime)..."
pip install -r "$PROJECT_DIR/requirements.txt"

info "Python 依赖安装完成"
pip list | grep -E "openwakeword|vosk|onnxruntime|openai|edge-tts|pyaudio|numpy" || true

# ============================================================
# Step 6: 下载 AI 模型
# ============================================================
step "Step 6/7: 下载 AI 模型"

MODELS_DIR="$PROJECT_DIR/models"
mkdir -p "$MODELS_DIR"

# 6a. Vosk 中文大模型 (1.3GB, 准确率更高)
VOSK_MODEL_DIR="$MODELS_DIR/vosk-model-cn-0.22"
if [ -d "$VOSK_MODEL_DIR" ]; then
    info "Vosk 模型已存在: $VOSK_MODEL_DIR"
else
    info "下载 Vosk 中文大模型 (1.3GB)..."
    cd "$MODELS_DIR"
    wget -q --show-progress \
        "https://alphacephei.com/vosk/models/vosk-model-cn-0.22.zip" \
        -O vosk-model.zip
    unzip -q vosk-model.zip
    rm vosk-model.zip
    cd "$PROJECT_DIR"
    info "Vosk 模型下载完成"
fi

# 6b. 唤醒词模型 (提示)
info ""
info "唤醒词模型需要自行训练或下载:"
info "  1. 使用 openWakeWord Colab 训练自定义唤醒词:"
info "     https://github.com/dscripka/openWakeWord"
info "  2. 将训练好的 .onnx 文件放到: $MODELS_DIR/openwakeword/"
info "  3. 更新 config/config.yaml 中的 wake_word.model_path"
info "  4. 暂时可以使用 demo 模式 (无唤醒词)"

mkdir -p "$MODELS_DIR/openwakeword"

# ============================================================
# Step 7: 配置 systemd 服务 (可选)
# ============================================================
step "Step 7/7: 配置开机自启动"

read -p "是否安装 systemd 服务实现开机自启动? [y/N] " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    SERVICE_FILE="/etc/systemd/system/smart-speaker.service"

    # 替换占位符
    sudo cp "$PROJECT_DIR/scripts/service/smart-speaker.service" "$SERVICE_FILE"
    sudo sed -i "s|/home/pi/smart_speaker|$PROJECT_DIR|g" "$SERVICE_FILE"
    sudo sed -i "s|/home/pi/smart_speaker_venv|$VENV_DIR|g" "$SERVICE_FILE"
    sudo sed -i "s|User=pi|User=$USER|g" "$SERVICE_FILE"
    sudo sed -i "s|Group=pi|Group=$USER|g" "$SERVICE_FILE"
    sudo sed -i "s|/run/user/1000|/run/user/$(id -u)|g" "$SERVICE_FILE"

    sudo systemctl daemon-reload
    sudo systemctl enable smart-speaker
    info "systemd 服务已安装并启用"
    info "启动服务: sudo systemctl start smart-speaker"
    info "查看日志: journalctl -u smart-speaker -f"
else
    info "跳过 systemd 服务安装"
fi

# ============================================================
# 完成
# ============================================================
echo ""
echo "╔══════════════════════════════════════════╗"
echo "║     🎉 安装完成!                         ║"
echo "╚══════════════════════════════════════════╝"
echo ""
echo "下一步:"
echo "  1. 设置环境变量:"
echo "     export DEEPSEEK_API_KEY='your-api-key'  # 必须"
echo "     export WEATHER_API_KEY='your-api-key'   # 可选 (天气技能)"
echo ""
echo "  2. 测试麦克风和扬声器:"
echo "     python src/main.py --list-devices"
echo ""
echo "  3. 训练/下载唤醒词模型 (或使用 demo 模式)"
echo ""
echo "  4. 启动:"
echo "     source ~/smart_speaker_venv/bin/activate"
echo "     python src/main.py"
echo ""
echo "  5. (可选) 使用 systemd 启动:"
echo "     sudo systemctl start smart-speaker"
echo "     journalctl -u smart-speaker -f"
echo ""
