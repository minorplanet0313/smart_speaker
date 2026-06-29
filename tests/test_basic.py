"""
基础功能测试

在开发机上运行, 不需要 Raspberry Pi 硬件:
    pytest tests/test_basic.py -v
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


class TestConfig:
    """配置加载测试"""

    def test_config_load(self):
        from src.utils.config import Config
        config = Config("config/config.yaml")
        assert config.get("general.name") == "小智音箱"
        assert config.get("audio.sample_rate") == 16000

    def test_config_defaults(self):
        from src.utils.config import Config
        config = Config("config/config.yaml")
        assert config.get("nonexistent.key", 42) == 42


class TestEventBus:
    """事件总线测试"""

    def test_subscribe_and_publish(self):
        from src.core.event_bus import Event, EventBus

        bus = EventBus()
        received = []

        def handler(data):
            received.append(data.get("msg"))

        bus.subscribe(Event.WAKE_WORD_DETECTED, handler)
        bus.publish(Event.WAKE_WORD_DETECTED, source="test", msg="hello")
        bus.publish(Event.WAKE_WORD_DETECTED, source="test", msg="world")

        assert received == ["hello", "world"]
        bus.clear()

    def test_unsubscribe(self):
        from src.core.event_bus import Event, EventBus

        bus = EventBus()
        received = []

        def handler(data):
            received.append(1)

        bus.subscribe(Event.WAKE_WORD_DETECTED, handler)
        bus.publish(Event.WAKE_WORD_DETECTED, source="test")
        bus.unsubscribe(Event.WAKE_WORD_DETECTED, handler)
        bus.publish(Event.WAKE_WORD_DETECTED, source="test")

        assert len(received) == 1
        bus.clear()


class TestStateMachine:
    """状态机测试"""

    def test_valid_transitions(self):
        from src.core.state_machine import State, StateMachine

        sm = StateMachine()
        assert sm.current_state == State.IDLE

        # IDLE → LISTENING (wake word)
        assert sm.transition(State.LISTENING) is True
        assert sm.current_state == State.LISTENING

        # LISTENING → THINKING
        assert sm.transition(State.THINKING) is True

        # THINKING → SPEAKING
        assert sm.transition(State.SPEAKING) is True

        # SPEAKING → IDLE
        assert sm.transition(State.IDLE) is True

    def test_invalid_transitions(self):
        from src.core.state_machine import State, StateMachine

        sm = StateMachine()

        # IDLE → SPEAKING (不合法)
        assert sm.transition(State.SPEAKING) is False
        assert sm.current_state == State.IDLE

    def test_mute_anytime(self):
        from src.core.state_machine import State, StateMachine

        sm = StateMachine()
        # 从任何状态都可以 MUTED
        assert sm.transition(State.MUTED) is True
        assert sm.transition(State.IDLE) is True

    def test_timeout_persists_after_trigger(self):
        import time
        from src.core.state_machine import State, StateMachine

        sm = StateMachine()
        triggered = []

        sm.set_timeout(State.LISTENING, 50, lambda: triggered.append(1))
        sm.transition(State.LISTENING)
        time.sleep(0.08)
        sm.check_timeouts()
        assert len(triggered) == 1

        sm.force_idle()
        sm.transition(State.LISTENING)
        time.sleep(0.08)
        sm.check_timeouts()
        assert len(triggered) == 2


class TestSentenceSplit:
    """分句工具测试"""

    def test_extract_sentences(self):
        from src.utils.sentence_split import extract_complete_sentences

        sentences, remainder = extract_complete_sentences("你好。世界！还有")
        assert sentences == ["你好。", "世界！"]
        assert remainder == "还有"


class TestRingBuffer:
    """环形缓冲区测试"""

    def test_append_and_consume(self):
        import numpy as np
        from src.audio.ring_buffer import AudioRingBuffer

        buf = AudioRingBuffer(max_samples=100)
        buf.append(np.ones(60, dtype=np.float32))
        buf.append(np.ones(60, dtype=np.float32))
        assert len(buf) == 100

        chunk = buf.consume(50)
        assert chunk is not None
        assert len(chunk) == 50


class TestPreprocessingCache:
    """预处理滤波器缓存测试"""

    def test_highpass_cache(self):
        import numpy as np
        from src.audio.preprocessing import highpass_filter, _HIGHPASS_SOS_CACHE

        _HIGHPASS_SOS_CACHE.clear()
        signal = np.random.randn(1600).astype(np.float32) * 0.1
        highpass_filter(signal, 16000, 80.0)
        highpass_filter(signal, 16000, 80.0)
        assert (16000, 80.0, 4) in _HIGHPASS_SOS_CACHE


class TestSkillKeywords:
    """技能关键词匹配测试"""

    def test_time_skill_no_false_positive(self):
        from src.skills.builtin.time_skill import TimeSkill

        skill = TimeSkill()
        assert skill.can_handle("现在几点了") is True
        assert skill.can_handle("时间旅行") is False


class TestConfigDebug:
    """配置项测试"""

    def test_debug_save_audio_default(self):
        from src.utils.config import Config

        config = Config("config/config.yaml")
        assert config.get("debug.save_audio") is False


class TestConversationContext:
    """对话上下文测试"""

    def test_basic_context(self):
        from src.llm.context import ConversationContext

        ctx = ConversationContext(max_history_rounds=5)
        conv_id = ctx.new_conversation()

        ctx.add_user_message("你好", conv_id)
        ctx.add_assistant_message("你好呀!", conv_id)

        messages = ctx.get_messages(conv_id)
        assert len(messages) == 2
        assert messages[0]["role"] == "user"
        assert messages[1]["role"] == "assistant"

    def test_trim_history(self):
        from src.llm.context import ConversationContext

        ctx = ConversationContext(max_history_rounds=2)
        conv_id = ctx.new_conversation()

        # 添加 5 轮对话 (10 条消息)
        for i in range(5):
            ctx.add_user_message(f"msg {i}")
            ctx.add_assistant_message(f"reply {i}")

        messages = ctx.get_messages(conv_id)
        # 应该被截断为 4 条消息 (2 轮)
        assert len(messages) == 4
        # 最早的消息被删除, 保留最新的
        assert messages[0]["content"] == "msg 3"


class TestSkillManager:
    """技能管理器测试"""

    def test_register_and_find(self):
        from src.skills.base import BaseSkill, SkillContext, SkillPriority, SkillResult
        from src.skills.skill_manager import SkillManager

        class MockSkill(BaseSkill):
            name = "mock"
            keywords = ["测试"]
            priority = SkillPriority.HIGH

            def can_handle(self, text):
                return "测试" in text

            def execute(self, text, context):
                return SkillResult(success=True, response_text="mock response")

        mgr = SkillManager()
        mgr.register(MockSkill())

        skill = mgr.find_handler("这是一个测试")
        assert skill is not None
        assert skill.name == "mock"

        # 不匹配
        skill = mgr.find_handler("其他内容")
        assert skill is None

        mgr.clear()


class TestAudioUtils:
    """音频工具测试"""

    def test_int16_float32_conversion(self):
        import numpy as np
        from src.audio.utils import int16_to_float32, float32_to_int16

        original = np.array([0, 16384, -16384, 32767, -32768], dtype=np.int16)
        float32 = int16_to_float32(original)
        restored = float32_to_int16(float32)

        # 应该近似还原 (±1 的量化误差)
        assert np.all(np.abs(original - restored) <= 1)

    def test_compute_rms(self):
        import numpy as np
        from src.audio.utils import compute_rms

        silence = np.zeros(1000, dtype=np.float32)
        assert compute_rms(silence) == 0.0

        sine = np.sin(np.linspace(0, 2 * np.pi, 1000)).astype(np.float32)
        rms = compute_rms(sine)
        assert 0.6 < rms < 0.8  # sin 的 RMS ≈ 0.707


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
