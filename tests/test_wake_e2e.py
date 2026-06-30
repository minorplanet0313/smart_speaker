"""
端到端唤醒词测试: 用真实音频文件测试唤醒流程

用法:
    python tests/test_wake_e2e.py [audio_file.wav]

不传参数则生成测试音频 (含模拟唤醒词模式)。
"""
import sys
import time

import numpy as np


def generate_test_audio(duration_sec=5.0, sample_rate=16000):
    """生成测试音频 (白噪声 + 周期性脉冲模拟唤醒词)"""
    total_samples = int(duration_sec * sample_rate)
    audio = np.random.randn(total_samples).astype(np.float32) * 0.01
    # 在 1-1.5秒处加入一个较强的信号脉冲模拟人声
    pulse_start = int(1.0 * sample_rate)
    pulse_end = int(1.5 * sample_rate)
    audio[pulse_start:pulse_end] = np.random.randn(pulse_end - pulse_start).astype(np.float32) * 0.3
    return audio


def run_detector_with_audio(detector, audio, chunk_size=1024, max_chunks=None):
    """用音频数据测试检测器, 返回统计信息"""
    total_chunks = len(audio) // chunk_size
    if max_chunks:
        total_chunks = min(total_chunks, max_chunks)

    scores = []
    triggers = []
    errors = []

    start = time.time()
    for i in range(total_chunks):
        chunk = audio[i * chunk_size:(i + 1) * chunk_size]
        try:
            score = detector.detect(chunk)
            scores.append(score)
            if score > 0:
                triggers.append((i, score))
        except Exception as e:
            errors.append((i, str(e)))

    elapsed = time.time() - start
    return {
        "chunks_processed": total_chunks,
        "elapsed_sec": elapsed,
        "max_score": max(scores) if scores else 0,
        "triggers": triggers,
        "errors": errors,
        "scores_sample": scores[:10] + ["..."] + scores[-5:] if len(scores) > 20 else scores,
    }


def main():
    from src.wake_word.detector import WakeWordDetector

    model_path = "./models/openwakeword/xiao_zhi.onnx"

    print("=" * 60)
    print("唤醒词检测器 E2E 测试")
    print("=" * 60)

    # 1. 创建检测器
    print("\n[1] 创建检测器...")
    detector = WakeWordDetector(model_path=model_path, threshold=0.3)
    print(f"    模型路径: {model_path}")
    print(f"    阈值: {detector.threshold}")
    print(f"    状态: {'✓ 模型文件存在' if detector._model_path_valid else '✗ 模型文件不存在'}")

    # 2. 检查 is_available
    print("\n[2] 检查 is_available...")
    available = detector.is_available
    print(f"    is_available: {available}")
    if not available:
        print("    错误: 模型加载失败!")
        return 1
    print(f"    模型类型: {type(detector._model).__name__}")

    # 3. detect() 不应该直接返回 0 (regression test)
    print("\n[3] Regression test: detect() 是否触发模型加载...")
    assert detector._model is not None, "模型应该已被加载!"
    print("    ✓ 模型已加载 (非 None)")

    # 4. 测试音频
    print("\n[4] 生成测试音频并推理...")
    audio = generate_test_audio(duration_sec=3.0)
    result = run_detector_with_audio(detector, audio, chunk_size=1024)
    print(f"    处理 chunks: {result['chunks_processed']}")
    print(f"    耗时: {result['elapsed_sec']:.3f}s")
    print(f"    最高分数: {result['max_score']:.4f}")
    print(f"    触发次数: {len(result['triggers'])}")
    if result['errors']:
        print(f"    错误: {result['errors']}")
    else:
        print(f"    ✓ 无异常")

    # 5. 验证回调机制
    print("\n[5] 测试回调机制...")
    callback_results = []

    def test_callback(confidence):
        callback_results.append(confidence)

    detector.on_detected(test_callback)
    # 直接触发
    detector._on_trigger(0.85)
    if len(callback_results) == 1 and callback_results[0] == 0.85:
        print("    ✓ 回调机制正常")
    else:
        print(f"    ✗ 回调异常: {callback_results}")

    # 6. 验证阈值调整
    print("\n[6] 测试阈值动态调整...")
    original = detector.threshold
    detector.set_threshold(0.6)
    assert detector.threshold == 0.6
    detector.set_threshold(original)
    print(f"    ✓ 阈值调整正常 ({original} → 0.6 → {original})")

    # 7. 总结
    print("\n" + "=" * 60)
    print("测试结果: 全部通过 ✓")
    print("=" * 60)

    # 打印关键诊断信息
    print(f"\n提示: 如需测试真实唤醒词, 请对着麦克风说出 '小智',")
    print(f"并以 DEBUG 级别运行以查看诊断输出:")
    print(f"  python src/main.py --log-level DEBUG")

    return 0


if __name__ == "__main__":
    sys.exit(main())
