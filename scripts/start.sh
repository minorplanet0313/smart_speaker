#!/usr/bin/env bash
# ============================================================
# Smart Speaker — 启动脚本
# ============================================================

set -e

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
VENV_DIR="$HOME/smart_speaker_venv"

# 激活虚拟环境
if [ -f "$VENV_DIR/bin/activate" ]; then
    source "$VENV_DIR/bin/activate"
else
    echo "错误: 虚拟环境未找到, 请先运行 scripts/install.sh"
    exit 1
fi

# 检查 API Key
if [ -z "$DEEPSEEK_API_KEY" ]; then
    if [ -f "$PROJECT_DIR/.env" ]; then
        source "$PROJECT_DIR/.env"
    fi
fi

if [ -z "$DEEPSEEK_API_KEY" ]; then
    echo "⚠️  警告: DEEPSEEK_API_KEY 未设置!"
    echo "   请创建 .env 文件或设置环境变量:"
    echo "   export DEEPSEEK_API_KEY='sk-...'"
    echo ""
    read -p "继续启动? (LLM 功能将不可用) [y/N] " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# 确保 PulseAudio 运行
if ! pulseaudio --check 2>/dev/null; then
    echo "启动 PulseAudio..."
    pulseaudio --start
fi

# 启动
cd "$PROJECT_DIR"
exec python src/main.py "$@"
