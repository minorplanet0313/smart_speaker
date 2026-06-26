#!/usr/bin/env python3
"""唤醒问题诊断脚本

快速检查各环节是否正常:
  - openWakeWord 安装
  - 模型文件存在
  - 麦克风设备
  - 唤醒词模型推理
  - 实时唤醒检测 (按 Ctrl+C 停止)
"""

import sys
import os
import time

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pyaudio


def check_import():
    """检查 openWakeWord 是否可导入"""
    print("=" * 50)
    print("1. 检查 openWakeWord 安装...")
    try:
        import openwakeword
        print(f"   ✅ openWakeWord {openwakeword.__version__ if hasattr(openwakeword, '__version__') else '(版本未知)'}")
        return True
    except ImportError as e:
        print(f"   ❌ 未安装: {e}")
        print("   修复: pip install openwakeword")
        return False


def check_models():
    """检查模型文件"""
    print("\n" + "=" * 50)
    print("2. 检查模型文件...")
    from pathlib import Path

    model_dir = Path("models/openwakeword")
    required = [
        "melspectrogram.onnx",
        "embedding_model.onnx",
    ]
    wake_models = list(model_dir.glob("*.onnx")) + list(model_dir.glob("*.tflite"))
    # 过滤掉辅助模型
    wake_models = [m for m in wake_models if m.name not in required]
    wake_models = [m for m in wake_models if "silero" not in m.name]

    all_ok = True
    for f in required:
        p = model_dir / f
        if p.exists():
            print(f"   ✅ {f} ({p.stat().st_size / 1024:.0f} KB)")
        else:
            print(f"   ❌ {f} 不存在 (路径: {p})")
            all_ok = False

    if wake_models:
        print(f"\n   唤醒词模型:")
        for m in wake_models:
            print(f"   📢 {m.name} ({m.stat().st_size / 1024:.0f} KB)")
    else:
        print(f"\n   ⚠️  未找到唤醒词模型文件 (*.onnx / *.tflite)")

    return all_ok and len(wake_models) > 0


def check_microphone():
    """检查麦克风设备"""
    print("\n" + "=" * 50)
    print("3. 检查音频输入设备...")

    pa = pyaudio.PyAudio()
    input_devices = []
    for i in range(pa.get_device_count()):
        info = pa.get_device_info_by_index(i)
        if info.get("maxInputChannels", 0) > 0:
            input_devices.append(info)
            default_marker = " [默认]" if i == pa.get_default_input_device_info().get("index") else ""
            print(f"   🎤 [{i}] {info['name']} "
                  f"(ch={info['maxInputChannels']}, "
                  f"rate={int(info.get('defaultSampleRate', 0))}){default_marker}")
    pa.terminate()

    if not input_devices:
        print("   ❌ 没有找到输入设备! 请检查 USB 麦克风连接")
        return False
    print(f"   ✅ 找到 {len(input_devices)} 个输入设备")
    return True


def test_model_inference():
    """测试模型推理"""
    print("\n" + "=" * 50)
    print("4. 测试唤醒词模型推理...")

    from openwakeword import Model
    from pathlib import Path

    model_dir = "models/openwakeword"

    try:
        # 加载模型
        model_path = f"{model_dir}/alexa_v0.1.onnx"
        if not Path(model_path).exists():
            print(f"   ❌ 模型文件不存在: {model_path}")
            return False

        m = Model(
            wakeword_models=[model_path],
            inference_framework="onnx",
        )
        print(f"   ✅ 模型加载成功")
        print(f"   可用模型: {list(m.models.keys())}")

        # 测试推理 (用随机噪声)
        print("   测试推理...")
        chunk = np.random.randn(1280).astype(np.float32) * 0.01  # 模拟静音
        predictions = m.predict(chunk)
        for name, score in predictions.items():
            print(f"   📊 {name}: {score:.4f} (随机噪声, 应接近 0)")
            if score > 0.5:
                print(f"   ⚠️  静音下置信度过高, 可能阈值太低或模型有问题")

        return True
    except Exception as e:
        print(f"   ❌ 模型推理失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def live_test():
    """实时唤醒词检测"""
    print("\n" + "=" * 50)
    print("5. 实时唤醒词检测")
    print("   " + "-" * 40)
    print("   🎤 开始实时监听...")
    print("   请对麦克风说话, 观察唤醒词检测情况")
    print("   当前模型: alexa_v0.1.onnx (检测词: \"Alexa\")")
    print("   配置阈值: 0.5")
    print("   按 Ctrl+C 退出")
    print("   " + "-" * 40)

    from openwakeword import Model
    import pyaudio

    model_path = "models/openwakeword/alexa_v0.1.onnx"

    try:
        m = Model(wakeword_models=[model_path], inference_framework="onnx")
    except Exception as e:
        print(f"   ❌ 模型加载失败: {e}")
        return

    pa = pyaudio.PyAudio()
    sample_rate = 16000
    chunk_size = 1280  # 80ms @ 16kHz

    stream = pa.open(
        format=pyaudio.paInt16,
        channels=1,
        rate=sample_rate,
        input=True,
        frames_per_buffer=chunk_size,
    )

    print(f"\n   模型: {list(m.models.keys())}")
    print(f"   采样率: {sample_rate}Hz, 帧大小: {chunk_size} samples")
    print(f"   开始监听...\n")

    last_print = 0
    try:
        while True:
            data = stream.read(chunk_size, exception_on_overflow=False)
            audio = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
            predictions = m.predict(audio)

            now = time.time()
            for name, score in predictions.items():
                if score > 0.1:  # 显示所有 > 0.1 的分数
                    marker = "🔥🔥🔥 唤醒!" if score > 0.5 else "📢 疑似"
                    print(f"   {marker} [{name}] confidence={score:.4f}")
                elif score > 0.01 and now - last_print > 3:
                    # 每 3 秒打印一次低置信度
                    print(f"   🔇 [{name}] confidence={score:.4f} (未达阈值)")
                    last_print = now

    except KeyboardInterrupt:
        print("\n   用户中断")
    finally:
        stream.stop_stream()
        stream.close()
        pa.terminate()


def main():
    print("🔍 Smart Speaker 唤醒诊断工具")
    print("=" * 50)

    checks = [
        ("openWakeWord 导入", check_import),
        ("模型文件", check_models),
        ("麦克风设备", check_microphone),
        ("模型推理", test_model_inference),
    ]

    results = {}
    for name, fn in checks:
        try:
            results[name] = fn()
        except Exception as e:
            print(f"\n   ❌ {name} 检查异常: {e}")
            import traceback
            traceback.print_exc()
            results[name] = False

    # 汇总
    print("\n" + "=" * 50)
    print("📋 诊断汇总:")
    all_pass = True
    for name, ok in results.items():
        status = "✅ 通过" if ok else "❌ 失败"
        print(f"   {status} - {name}")
        if not ok:
            all_pass = False

    if all_pass:
        print("\n   ✅ 所有基础检查通过!")
        print("\n   ⚠️  注意: 当前使用的是 alexa_v0.1.onnx 模型")
        print("   这意味着你只能说 \"Alexa\" (英文) 来唤醒")
        print("   如果你想用 \"小智小智\", 需要训练自定义模型:")
        print("   https://github.com/dscripka/openWakeWord#training-custom-models")
    else:
        print("\n   ❌ 部分检查未通过, 请根据上方提示修复")

    # 是否进入实时测试
    if all_pass:
        print("\n" + "-" * 40)
        try:
            ans = input("是否进入实时唤醒检测? [y/N]: ").strip().lower()
            if ans == 'y':
                live_test()
        except (EOFError, KeyboardInterrupt):
            pass


if __name__ == "__main__":
    main()
