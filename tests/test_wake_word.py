"""
唤醒词检测器测试

验证:
1. 模型延迟加载机制 (lazy_load 在 detect() 中被调用)
2. 调试计数器初始化 (不会 AttributeError)
3. 模型实际可加载 (is_available)
4. 无模型路径时的降级行为
"""
import os

import numpy as np
import pytest


class TestWakeWordDetectorInit:
    """测试 WakeWordDetector 初始化"""

    def test_init_attributes(self):
        """所有属性在 __init__ 中正确初始化"""
        from src.wake_word.detector import WakeWordDetector

        d = WakeWordDetector(model_path="./models/openwakeword/xiao_zhi.onnx")

        assert d.threshold == 0.5
        assert d._model is None
        assert d._model_loaded is False
        assert d._last_trigger_time == 0.0
        assert d._samples_per_chunk == 1280  # 80ms * 16kHz
        assert len(d._callbacks) == 0

    def test_debug_counters_initialized(self):
        """调试计数器在 __init__ 中初始化 (regression: AttributeError)"""
        from src.wake_word.detector import WakeWordDetector

        d = WakeWordDetector(model_path="./models/openwakeword/xiao_zhi.onnx")

        assert d._debug_sample_count == 0
        assert d._debug_rms_sum == 0.0
        assert d._debug_predict_count == 0
        assert d._debug_max_score == 0.0

    def test_init_no_model_path(self):
        """无模型路径时也能正常初始化"""
        from src.wake_word.detector import WakeWordDetector

        d = WakeWordDetector(model_path="")

        assert d._model_path_valid is False
        assert d._model is None


class TestWakeWordDetectorLazyLoad:
    """测试延迟加载机制"""

    def test_lazy_load_triggered_by_detect(self):
        """detect() 应触发 _lazy_load_model() — regression test for critical bug"""
        from src.wake_word.detector import WakeWordDetector

        d = WakeWordDetector(model_path="./models/openwakeword/xiao_zhi.onnx")

        # 调用 detect 前 model_loaded 应为 False
        assert d._model_loaded is False

        # 传入合法音频帧, detect() 内部应该调用 _lazy_load_model()
        audio = np.zeros(1024, dtype=np.float32)
        d.detect(audio)

        # 调用后 model_loaded 应为 True (即使加载失败也会标记为已尝试)
        assert d._model_loaded is True, (
            "detect() 未触发 _lazy_load_model()! "
            "这是关键 bug: 模型从未加载, 唤醒词永远无法触发"
        )

    def test_lazy_load_idempotent(self):
        """多次调用 _lazy_load_model 不会重复加载"""
        from src.wake_word.detector import WakeWordDetector

        d = WakeWordDetector(model_path="./models/openwakeword/xiao_zhi.onnx")
        d._lazy_load_model()
        first_model = d._model
        d._lazy_load_model()
        second_model = d._model

        assert first_model is second_model

    def test_detect_returns_float(self):
        """detect() 返回 float 类型的置信度"""
        from src.wake_word.detector import WakeWordDetector

        d = WakeWordDetector(model_path="./models/openwakeword/xiao_zhi.onnx")
        audio = np.zeros(1024, dtype=np.float32)
        score = d.detect(audio)

        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0


class TestWakeWordDetectorModelLoading:
    """测试实际模型加载"""

    @pytest.mark.skipif(
        not os.path.exists("./models/openwakeword/xiao_zhi.onnx"),
        reason="xiao_zhi.onnx 模型文件不存在",
    )
    def test_model_loads_with_real_model(self):
        """使用真实模型文件测试加载"""
        from src.wake_word.detector import WakeWordDetector

        d = WakeWordDetector(
            model_path="./models/openwakeword/xiao_zhi.onnx",
            threshold=0.3,
        )

        # is_available 应该触发加载并返回 True
        assert d.is_available is True, "模型文件存在但加载失败!"
        assert d._model is not None

    @pytest.mark.skipif(
        not os.path.exists("./models/openwakeword/xiao_zhi.onnx"),
        reason="xiao_zhi.onnx 模型文件不存在",
    )
    def test_detect_with_real_model(self):
        """使用真实模型测试 detect() 不会崩溃"""
        from src.wake_word.detector import WakeWordDetector

        d = WakeWordDetector(
            model_path="./models/openwakeword/xiao_zhi.onnx",
            threshold=0.3,
        )

        # 发送足够的静音帧以触发推理
        audio = np.zeros(1024, dtype=np.float32)
        for _ in range(10):
            score = d.detect(audio)
            assert isinstance(score, float)
            assert 0.0 <= score <= 1.0

    def test_detect_no_model_path_returns_zero(self):
        """无模型路径时 detect() 始终返回 0.0"""
        from src.wake_word.detector import WakeWordDetector

        d = WakeWordDetector(model_path="")
        audio = np.random.randn(1024).astype(np.float32) * 0.1

        for _ in range(10):
            score = d.detect(audio)
            assert score == 0.0, "无模型时应返回 0.0"


class TestWakeWordDetectorAudioFormat:
    """测试 detect() 传给 predict() 的音频格式"""

    def test_predict_receives_int16(self):
        """验证 predict() 收到的是 int16 而非 float32 数据"""
        from unittest.mock import MagicMock, ANY
        from src.wake_word.detector import WakeWordDetector

        d = WakeWordDetector(model_path="./models/openwakeword/xiao_zhi.onnx")
        d._lazy_load_model()

        # 用 mock 替换 predict 方法
        captured_inputs = []
        original_predict = d._model.predict
        d._model.predict = lambda x: (captured_inputs.append(x), {"test": 0.0})[1]

        try:
            # 发送 float32 音频
            audio = (np.sin(np.linspace(0, 2*np.pi*440*1280/16000, 1024)) * 0.3).astype(np.float32)
            # 需要填充到至少 1280 samples (2个 chunk)
            for _ in range(3):
                d.detect(audio)

            # 验证 predict 收到的是 int16
            assert len(captured_inputs) > 0, "predict() 未被调用"
            for inp in captured_inputs:
                assert inp.dtype == np.int16, (
                    f"predict() 应收到 int16, 实际收到 {inp.dtype}. "
                    f"float32 在 openWakeWord 内部会被截断为 0!"
                )
            print(f"✓ predict() 收到 {len(captured_inputs)} 次调用, dtype={captured_inputs[0].dtype}")
        finally:
            d._model.predict = original_predict


class TestWakeWordDetectorCallbacks:
    """测试回调机制"""

    def test_register_callback(self):
        from src.wake_word.detector import WakeWordDetector

        d = WakeWordDetector(model_path="")
        called = []

        def cb(conf):
            called.append(conf)

        d.on_detected(cb)
        assert len(d._callbacks) == 1

    def test_set_threshold_clamped(self):
        from src.wake_word.detector import WakeWordDetector

        d = WakeWordDetector(model_path="")
        d.set_threshold(1.5)
        assert d.threshold == 1.0
        d.set_threshold(-0.5)
        assert d.threshold == 0.0
        d.set_threshold(0.7)
        assert d.threshold == 0.7
