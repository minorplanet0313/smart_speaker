"""
GPIO 物理按键

支持:
- 静音/取消静音按键 (短按)
- 快捷操作按键 (短按: 报时, 长按: 关机等)
- 软件去抖动
"""

import threading
import time
from typing import Callable, Optional

from src.core.event_bus import Event, EventBus
from src.utils.logger import get_logger

logger = get_logger(__name__)


class Button:
    """
    物理按键

    特性:
    - 软件去抖动 (50ms)
    - 长按检测 (可配置)
    - 回调模式
    """

    def __init__(
        self,
        pin: int,
        name: str = "button",
        pull_up: bool = True,
        debounce_ms: int = 50,
        long_press_ms: int = 3000,
    ):
        self.pin = pin
        self.name = name
        self.pull_up = pull_up
        self.debounce_ms = debounce_ms
        self.long_press_ms = long_press_ms

        self._gpio_available = False
        self._last_press_time = 0.0
        self._press_start_time: Optional[float] = None
        self._long_press_triggered = False
        self._running = False
        self._thread: Optional[threading.Thread] = None

        self._on_short_press: Optional[Callable[[], None]] = None
        self._on_long_press: Optional[Callable[[], None]] = None

        self._init_gpio()

    def _init_gpio(self) -> None:
        """初始化 GPIO 按键"""
        try:
            import RPi.GPIO as GPIO
            GPIO.setmode(GPIO.BCM)
            GPIO.setwarnings(False)

            pull = GPIO.PUD_UP if self.pull_up else GPIO.PUD_DOWN
            GPIO.setup(self.pin, GPIO.IN, pull_up_down=pull)

            self._gpio_available = True
            logger.info(f"按键初始化完成: {self.name} (GPIO{self.pin})")
        except ImportError:
            logger.info(f"RPi.GPIO 不可用, 按键 {self.name} 降级为无效")
        except Exception as e:
            logger.warning(f"GPIO 初始化失败: {e}")

    def on_short_press(self, callback: Callable[[], None]) -> None:
        """设置短按回调"""
        self._on_short_press = callback

    def on_long_press(self, callback: Callable[[], None]) -> None:
        """设置长按回调"""
        self._on_long_press = callback

    def start(self) -> None:
        """开始监听按键"""
        if not self._gpio_available:
            return

        self._running = True
        self._thread = threading.Thread(
            target=self._poll_loop,
            daemon=True,
            name=f"button-{self.name}",
        )
        self._thread.start()
        logger.info(f"按键监听已启动: {self.name}")

    def stop(self) -> None:
        """停止监听"""
        self._running = False
        if self._thread:
            self._thread.join(timeout=1)

    def _poll_loop(self) -> None:
        """轮询按键状态"""
        import RPi.GPIO as GPIO

        while self._running:
            try:
                # 读取按键状态 (上拉模式: 按下 = LOW)
                state = GPIO.input(self.pin)
                pressed = (state == GPIO.LOW) if self.pull_up else (state == GPIO.HIGH)

                now = time.time()

                if pressed:
                    if self._press_start_time is None:
                        # 去抖动
                        if now - self._last_press_time > self.debounce_ms / 1000.0:
                            self._press_start_time = now
                            self._long_press_triggered = False
                            logger.debug(f"按键按下: {self.name}")

                    # 检查长按
                    if (self._press_start_time and
                            not self._long_press_triggered and
                            now - self._press_start_time > self.long_press_ms / 1000.0):
                        self._long_press_triggered = True
                        logger.info(f"长按触发: {self.name}")
                        if self._on_long_press:
                            self._on_long_press()
                else:
                    if self._press_start_time is not None:
                        press_duration = now - self._press_start_time
                        if not self._long_press_triggered:
                            logger.info(f"短按触发: {self.name} "
                                        f"(duration={press_duration*1000:.0f}ms)")
                            if self._on_short_press:
                                self._on_short_press()
                        self._press_start_time = None
                        self._last_press_time = now

                time.sleep(0.05)  # 50ms 轮询

            except Exception as e:
                logger.error(f"按键读取异常: {e}")
                time.sleep(0.5)

    def cleanup(self) -> None:
        """清理"""
        self.stop()
        logger.debug(f"按键 {self.name} 已清理")
