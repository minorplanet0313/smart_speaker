"""
GPIO LED 状态指示灯

通过不同颜色/闪烁模式指示设备状态:
- 绿色常亮: IDLE (等待唤醒)
- 蓝色呼吸: LISTENING (正在听)
- 紫色闪烁: THINKING (处理中)
- 绿色闪烁: SPEAKING (播放中)
- 红色常亮: MUTED (静音)
- 红色闪烁: ERROR

支持:
- 单色 LED (亮/灭/闪烁)
- RGB LED (颜色切换)
- 无 LED (纯日志输出, 在开发机/无 GPIO 环境使用)
"""

import threading
import time
from enum import Enum
from typing import Optional

from src.utils.logger import get_logger

logger = get_logger(__name__)


class LEDMode(Enum):
    OFF = "off"
    ON = "on"
    SLOW_BLINK = "slow_blink"     # 1秒周期
    FAST_BLINK = "fast_blink"     # 0.3秒周期
    BREATHING = "breathing"       # 呼吸效果 (仅 RGB)


class LEDStatus:
    """LED 状态"""

    def __init__(self, mode: LEDMode = LEDMode.OFF):
        self.mode = mode
        self.color = "green"  # 仅 RGB LED 有效
        self.brightness = 1.0


class LEDController:
    """
    LED 控制器

    支持单色 LED 和 RGB LED
    如果 GPIO 不可用 (如在开发机上), 降级为控制台日志
    """

    def __init__(
        self,
        pin: int = 17,
        led_type: str = "single",  # "single" | "rgb"
        use_pwm: bool = True,
    ):
        self.pin = pin
        self.led_type = led_type
        self.use_pwm = use_pwm

        self._gpio_available = False
        self._pwm = None
        self._blink_thread: Optional[threading.Thread] = None
        self._blink_running = False
        self._current_status = LEDStatus()

        self._init_gpio()

    def _init_gpio(self) -> None:
        """初始化 GPIO"""
        try:
            import RPi.GPIO as GPIO
            GPIO.setmode(GPIO.BCM)
            GPIO.setwarnings(False)

            if self.led_type == "single":
                GPIO.setup(self.pin, GPIO.OUT)
                self._gpio_available = True
                logger.info(f"单色 LED 初始化完成: GPIO{self.pin}")
            elif self.led_type == "rgb":
                # RGB LED 需要 3 个 GPIO 引脚 (R, G, B)
                # 这里简化: 假设 R=pin, G=pin+1, B=pin+2
                for p in [self.pin, self.pin + 1, self.pin + 2]:
                    GPIO.setup(p, GPIO.OUT)
                if self.use_pwm:
                    self._pwm = {
                        'r': GPIO.PWM(self.pin, 100),
                        'g': GPIO.PWM(self.pin + 1, 100),
                        'b': GPIO.PWM(self.pin + 2, 100),
                    }
                    for p in self._pwm.values():
                        p.start(0)
                self._gpio_available = True
                logger.info(f"RGB LED 初始化完成: GPIO{self.pin}-{self.pin+2}")
        except ImportError:
            logger.info("RPi.GPIO 不可用, LED 降级为日志输出")
            self._gpio_available = False
        except Exception as e:
            logger.warning(f"GPIO 初始化失败: {e}, LED 降级为日志输出")
            self._gpio_available = False

    def set_status(self, status: LEDStatus) -> None:
        """设置 LED 状态"""
        self._current_status = status

        if not self._gpio_available:
            # 日志输出模拟
            logger.debug(f"[LED] {status.mode.value} ({status.color})")
            return

        # 停止之前的闪烁
        self._stop_blink()

        if status.mode == LEDMode.OFF:
            self._led_off()
        elif status.mode == LEDMode.ON:
            self._led_on()
        elif status.mode in (LEDMode.SLOW_BLINK, LEDMode.FAST_BLINK):
            interval = 0.5 if status.mode == LEDMode.SLOW_BLINK else 0.15
            self._start_blink(interval)
        elif status.mode == LEDMode.BREATHING:
            self._start_breathing()

    def _led_on(self) -> None:
        try:
            import RPi.GPIO as GPIO
            if self.led_type == "single":
                GPIO.output(self.pin, GPIO.HIGH)
            elif self.led_type == "rgb" and self._pwm:
                self._set_rgb_color(self._current_status.color)
        except Exception:
            pass

    def _led_off(self) -> None:
        try:
            import RPi.GPIO as GPIO
            if self.led_type == "single":
                GPIO.output(self.pin, GPIO.LOW)
            elif self.led_type == "rgb" and self._pwm:
                for p in self._pwm.values():
                    p.ChangeDutyCycle(0)
        except Exception:
            pass

    def _start_blink(self, interval: float) -> None:
        self._blink_running = True
        self._blink_thread = threading.Thread(
            target=self._blink_loop,
            args=(interval,),
            daemon=True,
        )
        self._blink_thread.start()

    def _blink_loop(self, interval: float) -> None:
        import RPi.GPIO as GPIO
        while self._blink_running:
            GPIO.output(self.pin, GPIO.HIGH)
            time.sleep(interval)
            GPIO.output(self.pin, GPIO.LOW)
            time.sleep(interval)

    def _start_breathing(self) -> None:
        """呼吸灯效果 (PWM 渐亮渐暗)"""
        if not self._pwm or self.led_type != "rgb":
            self._start_blink(1.0)  # 降级为慢闪
            return

        self._blink_running = True
        self._blink_thread = threading.Thread(
            target=self._breathing_loop,
            daemon=True,
        )
        self._blink_thread.start()

    def _breathing_loop(self) -> None:
        """PWM 呼吸效果"""
        try:
            while self._blink_running:
                # 渐亮
                for dc in range(0, 101, 2):
                    if not self._blink_running:
                        break
                    self._pwm['g'].ChangeDutyCycle(dc)
                    time.sleep(0.02)
                # 渐暗
                for dc in range(100, -1, -2):
                    if not self._blink_running:
                        break
                    self._pwm['g'].ChangeDutyCycle(dc)
                    time.sleep(0.02)
        except Exception:
            pass

    def _stop_blink(self) -> None:
        self._blink_running = False
        if self._blink_thread:
            self._blink_thread.join(timeout=0.5)
            self._blink_thread = None

    def _set_rgb_color(self, color: str) -> None:
        """设置 RGB 颜色 (简化)"""
        colors = {
            "red":    (100, 0, 0),
            "green":  (0, 100, 0),
            "blue":   (0, 0, 100),
            "purple": (50, 0, 50),
            "yellow": (100, 100, 0),
            "white":  (100, 100, 100),
        }
        rgb = colors.get(color, (0, 100, 0))
        if self._pwm:
            self._pwm['r'].ChangeDutyCycle(rgb[0])
            self._pwm['g'].ChangeDutyCycle(rgb[1])
            self._pwm['b'].ChangeDutyCycle(rgb[2])

    def cleanup(self) -> None:
        """清理 GPIO 资源"""
        self._stop_blink()
        if self._gpio_available:
            try:
                import RPi.GPIO as GPIO
                GPIO.cleanup()
            except Exception:
                pass
        logger.debug("LED 已清理")
