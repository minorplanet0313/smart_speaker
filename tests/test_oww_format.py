"""验证 openWakeWord 内部的 float32 截断问题"""
import numpy as np


class TestOpenWakeWordFormat:
    """测试 openWakeWord 内部音频格式处理"""

    def test_float32_truncation_in_pipeline(self):
        """float32 [-1,1] → list → np.array → astype(int16): 截断为 0"""
        # 模拟检测器传入的 float32 音频 (正弦波, 0.5 振幅)
        float_audio = (np.sin(np.linspace(0, 2*np.pi*440*1280/16000, 1280)) * 0.5).astype(np.float32)

        # 模拟 openWakeWord 的 _buffer_raw_data: x.tolist()
        as_list = float_audio.tolist()

        # 模拟 _get_melspectrogram 第194行: np.array(list).astype(np.int16)
        x_int16 = np.array(as_list).astype(np.int16)

        # float32 值在 [-1, 1] 范围, astype(int16) 直接截断
        # 0.5 → 0, -0.3 → 0, 只有 1.0 → 1
        rms = float(np.sqrt(np.mean(x_int16.astype(float)**2)))
        max_val = int(x_int16.max())
        min_val = int(x_int16.min())

        print(f"float32→int16 转换结果: max={max_val}, min={min_val}, rms={rms:.2f}")
        print(f"  (原始音频 RMS 约 0.35, 16-bit 应有 RMS ~11500)")

        # RMS 应该远小于正确的 16-bit RMS (~11500)
        assert rms < 10, f"预期截断后接近静音 (RMS<10), 实际 RMS={rms:.2f}"

    def test_int16_preserved_in_pipeline(self):
        """int16 → list → np.array → astype(int16): 完整保留"""
        float_audio = (np.sin(np.linspace(0, 2*np.pi*440*1280/16000, 1280)) * 0.5).astype(np.float32)
        int16_audio = (float_audio * 32767).clip(-32768, 32767).astype(np.int16)

        # openWakeWord 内部路径
        as_list = int16_audio.tolist()
        x_int16 = np.array(as_list).astype(np.int16)

        rms = float(np.sqrt(np.mean(x_int16.astype(float)**2)))
        print(f"int16→int16 转换结果: rms={rms:.2f}")

        assert rms > 1000, f"预期保留音频能量 (RMS>1000), 实际 RMS={rms:.2f}"

    def test_fix_conversion(self):
        """修复: 在 detector.py 中 predict() 前转 int16"""
        float_audio = (np.sin(np.linspace(0, 2*np.pi*440*1280/16000, 1280)) * 0.5).astype(np.float32)

        # 修复: 转换为 int16
        int16_audio = (float_audio * 32767).clip(-32768, 32767).astype(np.int16)

        # 验证正确性
        rms = float(np.sqrt(np.mean(int16_audio.astype(float)**2)))
        print(f"修复后转换: rms={rms:.2f}")
        assert rms > 1000
