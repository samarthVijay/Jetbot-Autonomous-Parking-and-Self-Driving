import time
import math
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("PCA9685")

# PCA9685 Register Addresses
PCA9685_MODE1 = 0x00
PCA9685_MODE2 = 0x01
PCA9685_SUBADR1 = 0x02
PCA9685_SUBADR2 = 0x03
PCA9685_SUBADR3 = 0x04
PCA9685_PRESCALE = 0xFE
LED0_ON_L = 0x06
LED0_ON_H = 0x07
LED0_OFF_L = 0x08
LED0_OFF_H = 0x09
ALL_LED_ON_L = 0xFA
ALL_LED_ON_H = 0xFB
ALL_LED_OFF_L = 0xFC
ALL_LED_OFF_H = 0xFD

# Mode 1 Bits
RESTART = 0x80
SLEEP = 0x10
ALLCALL = 0x01
INVRT = 0x10
OUTDRV = 0x04


class PCA9685:
    """
    Low-level Python driver for the PCA9685 16-Channel 12-Bit PWM IC.
    Communicates via Linux I2C bus (/dev/i2c-1) using smbus2 or fallback mock.
    """
    def __init__(self, bus_num=1, address=0x60):
        self.bus_num = bus_num
        self.address = address
        self.bus = None
        self.is_mock = False

        try:
            import smbus2
            self.bus = smbus2.SMBus(self.bus_num)
            logger.info(f"Connected to I2C bus {bus_num} at address 0x{address:02X}")
            self._reset()
        except Exception as e:
            logger.warning(f"Could not open I2C bus {bus_num} (Error: {e}). Running in MOCK mode.")
            self.is_mock = True

    def _write_reg(self, reg, value):
        if not self.is_mock and self.bus:
            self.bus.write_byte_data(self.address, reg, value & 0xFF)

    def _read_reg(self, reg):
        if not self.is_mock and self.bus:
            return self.bus.read_byte_data(self.address, reg)
        return 0

    def _reset(self):
        """Reset PCA9685 to default state."""
        self._write_reg(PCA9685_MODE1, ALLCALL)
        self._write_reg(PCA9685_MODE2, OUTDRV)
        time.sleep(0.005)

    def set_pwm_freq(self, freq_hz=50):
        """
        Sets the PWM frequency in Hz (typically 50Hz to 1000Hz).
        Formula: prescale = round(25MHz / (4096 * freq_hz)) - 1
        """
        prescaleval = 25000000.0    # 25MHz internal clock
        prescaleval /= 4096.0       # 12-bit resolution
        prescaleval /= float(freq_hz)
        prescaleval -= 1.0
        prescale = math.floor(prescaleval + 0.5)

        if self.is_mock:
            logger.info(f"[MOCK] PWM frequency set to {freq_hz} Hz (prescale: {prescale})")
            return

        oldmode = self._read_reg(PCA9685_MODE1)
        newmode = (oldmode & 0x7F) | SLEEP  # Sleep mode to change prescaler
        self._write_reg(PCA9685_MODE1, newmode)
        self._write_reg(PCA9685_PRESCALE, int(prescale))
        self._write_reg(PCA9685_MODE1, oldmode)
        time.sleep(0.005)
        self._write_reg(PCA9685_MODE1, oldmode | RESTART)

    def set_motor_speeds(self, left_speed: float, right_speed: float):
        """
        Sets left and right motor speeds [-1.0 to 1.0].
        Channel mapping (Adafruit / Waveshare Motor HAT):
          Left:  PWM=8,  IN1=10, IN2=9
          Right: PWM=13, IN1=12, IN2=11
        """
        # --- Left Motor ---
        l_duty = min(max(abs(left_speed), 0.0), 1.0)
        l_pwm = int(l_duty * 4095)
        if left_speed > 0:
            self.set_pwm(10, 0, 4096)
            self.set_pwm(9, 4096, 0)
        elif left_speed < 0:
            self.set_pwm(10, 4096, 0)
            self.set_pwm(9, 0, 4096)
        else:
            self.set_pwm(10, 0, 4096)
            self.set_pwm(9, 0, 4096)
            l_pwm = 0
        self.set_pwm(8, 0, l_pwm)

        # --- Right Motor ---
        r_duty = min(max(abs(right_speed), 0.0), 1.0)
        r_pwm = int(r_duty * 4095)
        if right_speed > 0:
            self.set_pwm(12, 0, 4096)
            self.set_pwm(11, 4096, 0)
        elif right_speed < 0:
            self.set_pwm(12, 4096, 0)
            self.set_pwm(11, 0, 4096)
        else:
            self.set_pwm(12, 0, 4096)
            self.set_pwm(11, 0, 4096)
            r_pwm = 0
        self.set_pwm(13, 0, r_pwm)

    def set_pwm(self, channel, on_tick, off_tick):
        """
        Sets the ON and OFF 12-bit counter ticks (0 to 4095) for a channel (0-15).
        """
        if channel < 0 or channel > 15:
            raise ValueError("Channel must be between 0 and 15")

        if self.is_mock:
            return

        self._write_reg(LED0_ON_L + 4 * channel, on_tick & 0xFF)
        self._write_reg(LED0_ON_H + 4 * channel, (on_tick >> 8) & 0xFF)
        self._write_reg(LED0_OFF_L + 4 * channel, off_tick & 0xFF)
        self._write_reg(LED0_OFF_H + 4 * channel, (off_tick >> 8) & 0xFF)

    def set_duty_cycle(self, channel, duty_cycle):
        """
        Convenience function: set channel duty cycle from 0.0 (0%) to 1.0 (100%).
        """
        duty_cycle = max(0.0, min(1.0, duty_cycle))
        off_tick = int(duty_cycle * 4095)
        self.set_pwm(channel, 0, off_tick)

    def stop_all(self):
        """Turns off all channels."""
        self.set_pwm_freq(50)
        for ch in range(16):
            self.set_pwm(ch, 0, 0)
        logger.info("PCA9685 all channels stopped.")
