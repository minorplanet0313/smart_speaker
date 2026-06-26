#!/usr/bin/env python3
"""实时唤醒词检测测试 - 绕过主程序, 直接测试麦克风 → openWakeWord 流水线

这个脚本完全独立于 Smart Speaker 主程序, 帮助你隔离问题。
用法:
    python scripts/test_wake_live.py              # 默认: alexa 模型, 阈值 0.3
    python scripts/test_wake_live.py --low        # 极低阈值 (0.05), 显示所有分数
    python scripts/test_wake_live.py --list       # 列出音频设备
"""

import sys
import os
import time
import argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    parser = argparse.ArgumentParser(description="实时唤醒词检测测试")
    parser.add_argument("--low", action="store_true", help="极低阈值模式 (显示所有检测分数)")
    parser.add_argument("--list", action="store_true", help="列出音频设备")
    parser.add_argument("--threshold", type=float, default=0.3, help="检测阈值 (默认 0.3)")
    parser.add_argument("--model", default="models/openwakeword/alexa_v0.1.onnx", help="模型路径")
    parser.add_argument("--device", type=int, default=None, help="音频输入设备索引")
    args = parser.parse_args()

    if args.list:
        import pyaudio
        pa = pyaudio.PyAudio()
        print("\n🎤 音频输入设备:")
        for i in range(pa.get_device_count()):
            info = pa.get_device_info_by_index(i)
            if info.get("maxInputChannels", 0) > 0:
                mark = " [默认]" if i == pa.get_default_input_device_info().get("index") else ""
                print(f"  [{i}] {info['name']} (ch={info['maxInputChannels']}){mark}")
        pa.terminate()
        return

    # 加载模型
    print("加载唤醒词模型...")
    from openwakeword import Model
    model_path = args.model
    if not os.path.exists(model_path):
        print(f"❌ 模型不存在: {model_path}")
        sys.exit(1)

    m = Model(wakeword_models=[model_path], inference_framework="onnx")
    model_name = list(m.models.keys())[0]
    print(f"✅ 模型加载完成: {model_name}")
    print(f"   阈值: {args.threshold}")

    # 打开麦克风
    import pyaudio
    pa = pyaudio.PyAudio()

    device_index = args.device
    if device_index is None:
        device_index = pa.get_default_input_device_info().get("index")

    sample_rate = 16000
    chunk_size = 1024  # 与 AudioCapture 一致

    stream = pa.open(
        format=pyaudio.paInt16,
        channels=1,
        rate=sample_rate,
        input=True,
        input_device_index=device_index,
        frames_per_buffer=chunk_size,
    )
    print(f"🎤 麦克风已打开: [{device_index}] (chunk={chunk_size}, rate={sample_rate})")
    print(f"💡 请对麦克风说 '{model_name}' (英文)...")
    print(f"   按 Ctrl+C 退出\n")

    # 模拟 AudioCapture + WakeWordDetector 的流水线
    audio_buffer = np.array([], dtype=np.float32)
    samples_per_chunk = 1280  # 80ms @ 16kHz (openWakeWord 期望)

    frame_count = 0
    predict_count = 0
    max_score = 0.0
    total_audio = 0.0

    print(f"{'时间':>10s} {'RMS':>8s} {'分数':>8s} {'判定':>6s}")
    print("-" * 40)

    try:
        while True:
            # 从麦克风读取 (与 AudioCapture 完全一致)
            data = stream.read(chunk_size, exception_on_overflow=False)
            audio_frame = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0

            # 计算音频能量
            rms = float(np.sqrt(np.mean(np.square(audio_frame))))
            total_audio += len(audio_frame) / sample_rate
            frame_count += 1

            # 累积缓冲 (与 WakeWordDetector.detect 完全一致)
            audio_buffer = np.append(audio_buffer, audio_frame)

            if len(audio_buffer) >= samples_per_chunk:
                chunk = audio_buffer[:samples_per_chunk]
                audio_buffer = audio_buffer[samples_per_chunk:]

                predictions = m.predict(chunk)
                predict_count += 1

                for name, score in predictions.items():
                    if score > max_score:
                        max_score = score

                    detected = score > args.threshold or (args.low and score > 0.01)
                    if detected:
                        elapsed = total_audio
                        marker = "🔥🔥🔥 唤醒!" if score > args.threshold else "📢"
                        print(f"{elapsed:>8.1f}s {rms:>8.4f} {score:>8.4f} {marker}")
                    elif frame_count % 50 == 0:  # 每 ~3 秒报告一次状态
                        elapsed = total_audio
                        print(f"{elapsed:>8.1f}s {rms:>8.4f} {score:>8.4f} (max={max_score:.4f})")

    except KeyboardInterrupt:
        print(f"\n\n--- 统计 ---")
        print(f"   总录音时长: {total_audio:.1f}s")
        print(f"   接收帧数: {frame_count} (每帧 {chunk_size} samples)")
        print(f"   推理次数: {predict_count}")
        print(f"   最高分数: {max_score:.4f}")
        print(f"   音频 RMS 均值: {'(见上方)'}")
        if max_score < args.threshold:
            print(f"\n   ⚠️  最高分数 {max_score:.4f} 低于阈值 {args.threshold}")
            if max_score < 0.01:
                print(f"   🔴 几乎全是零分! 可能原因:")
                print(f"      1. 麦克风未插入或静音")
                print(f"      2. 系统选择了错误的输入设备")
                print(f"      3. 音频格式不匹配 (检查采样率)")
                print(f"   💡 尝试: python scripts/test_wake_live.py --list 查看设备")
                print(f"   💡 尝试: python scripts/test_wake_live.py --device <索引>")
    finally:
        stream.stop_stream()
        stream.close()
        pa.terminate()
        print("✅ 测试结束")


if __name__ == "__main__":
    main()
