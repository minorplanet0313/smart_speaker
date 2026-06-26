#!/usr/bin/env python3
"""
Smart Speaker — 智能语音助手

入口文件

用法:
    python src/main.py                          # 使用默认配置启动
    python src/main.py --config config/config.yaml
    python src/main.py --list-devices           # 列出音频设备
    python src/main.py --list-voices            # 列出 Edge TTS 语音
    python src/main.py --test-asr audio.wav     # 测试 ASR
"""

import argparse
import sys
from pathlib import Path

# 将 src 目录添加到 Python 路径
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.core.engine import SmartSpeakerEngine
from src.utils.logger import setup_logger


def parse_args():
    parser = argparse.ArgumentParser(
        description="Smart Speaker — 智能语音助手",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  %(prog)s                                   使用默认配置
  %(prog)s --config my_config.yaml           使用自定义配置
  %(prog)s --list-devices                    列出音频设备
  %(prog)s --list-voices                     列出可用的 TTS 语音
  %(prog)s --test-asr audio.wav              测试语音识别
        """,
    )

    parser.add_argument(
        "--config", "-c",
        default="config/config.yaml",
        help="配置文件路径 (默认: config/config.yaml)",
    )
    parser.add_argument(
        "--log-level", "-l",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default=None,
        help="日志级别 (覆盖配置文件)",
    )

    # 工具命令
    parser.add_argument(
        "--list-devices",
        action="store_true",
        help="列出所有音频输入/输出设备",
    )
    parser.add_argument(
        "--list-voices",
        action="store_true",
        help="列出 Edge TTS 可用的中文语音",
    )
    parser.add_argument(
        "--test-asr",
        metavar="AUDIO_FILE",
        help="测试 ASR 识别 (指定音频文件)",
    )
    parser.add_argument(
        "--stdin", "--text",
        action="store_true",
        dest="stdin_mode",
        help="键盘输入模式 (输入文字代替语音, 跳过唤醒和 ASR)",
    )
    parser.add_argument(
        "--keyboard-wake", "--kw",
        action="store_true",
        dest="keyboard_wake",
        help="键盘唤醒模式 (按回车唤醒, 然后用语音对话)",
    )

    return parser.parse_args()


def cmd_list_devices():
    """列出音频设备"""
    from src.audio.capture import AudioCapture
    from src.audio.player import AudioPlayer

    print("\n🎤 输入设备 (麦克风):")
    print("-" * 60)
    capture = AudioCapture()
    for dev in capture.input_devices:
        print(f"  [{dev['index']}] {dev['name']}")
        print(f"       channels={dev['channels']}, sample_rate={dev['sample_rate']}")

    print("\n🔊 输出设备 (扬声器):")
    print("-" * 60)
    player = AudioPlayer()
    for dev in player.output_devices:
        print(f"  [{dev['index']}] {dev['name']}")
        print(f"       channels={dev['channels']}")


def cmd_list_voices():
    """列出可用的 TTS 语音"""
    from src.tts.edge_tts import EdgeTTS

    print("\n🎙️  Edge TTS 中文语音:")
    print("-" * 60)
    voices = EdgeTTS.list_voices()
    for v in voices:
        print(f"  {v['name']:30s} {v['gender']:10s} {v['friendly_name']}")


def cmd_test_asr(audio_file: str):
    """测试 ASR 识别"""
    from src.asr.vosk_asr import VoskASR
    from src.utils.config import get_config

    config = get_config()
    model_path = config.get("asr.vosk.model_path", "models/vosk-model-cn-0.22")

    print(f"\n🎤 测试 ASR: {audio_file}")
    print(f"   模型路径: {model_path}")
    print("-" * 60)

    asr = VoskASR(model_path=model_path)
    if not asr.is_available:
        print("❌ Vosk 模型不可用, 请先下载模型")
        print("   wget https://alphacephei.com/vosk/models/vosk-model-cn-0.22.zip")
        return

    import time
    start = time.time()
    result = asr.transcribe_file(audio_file)
    elapsed = time.time() - start

    print(f"   识别结果: \"{result.text}\"")
    print(f"   置信度:   {result.confidence:.3f}")
    print(f"   耗时:     {elapsed:.2f}s")


def main():
    args = parse_args()

    # 工具命令 (不需要启动引擎)
    if args.list_devices:
        cmd_list_devices()
        return
    if args.list_voices:
        cmd_list_voices()
        return
    if args.test_asr:
        cmd_test_asr(args.test_asr)
        return

    # 启动引擎
    print("""
╔══════════════════════════════════════════╗
║         🎙️  Smart Speaker               ║
║         智能语音助手 v0.1.0               ║
╚══════════════════════════════════════════╝
    """)

    # 初始化日志
    log_level = args.log_level or "INFO"
    setup_logger(level=log_level)

    # 创建并启动引擎
    engine = SmartSpeakerEngine(config_path=args.config)
    engine.setup()

    if args.stdin_mode:
        # 键盘输入模式: 跳过唤醒词和语音识别, 直接输入文字
        engine.enable_stdin_mode()
    elif args.keyboard_wake:
        # 键盘唤醒模式: 键盘触发唤醒, 语音对话
        engine.enable_keyboard_wake_mode()
    else:
        wake_word = engine.config.get("general.wake_word", "小智小智")
        print(f"\n✅ 初始化完成, 唤醒词: {wake_word}")
        print(f"💡 说 '{wake_word}' 来唤醒我")
        print(f"💡 或使用 python src/main.py --stdin 进入文本输入模式")
        print(f"💡 或使用 python src/main.py --keyboard-wake 进入键盘唤醒模式\n")

    try:
        engine.run_forever()
    except KeyboardInterrupt:
        print("\n👋 再见!")
    except Exception as e:
        print(f"\n❌ 严重错误: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
