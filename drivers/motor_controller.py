import time
import logging
from .c_driver.python_wrapper import CPCA9685Driver
from .pca9685_i2c import PCA9685

logger = logging.getLogger("MotorController")


class MotorController:
    """
    Differential drive motor controller wrapping PCA9685 + TB6612FNG H-Bridge.
    Replaces jetbot.Robot with direct hardware register control and differential kinematics.
    """
    def __init__(self, i2c_bus=1, i2c_address=0x60, left_motor_channels=(8, 10, 9), right_motor_channels=(13, 12, 11)):
        """
        :param left_motor_channels: (PWM_channel, IN1_channel, IN2_channel) for left motor (Waveshare: 8, 10, 9)
        :param right_motor_channels: (PWM_channel, IN1_channel, IN2_channel) for right motor (Waveshare: 13, 12, 11)
        """
        # Try native C driver first for zero-overhead hardware I2C
        self.c_driver = CPCA9685Driver(i2c_device=f"/dev/i2c-{i2c_bus}", address=i2c_address)
        if not self.c_driver.is_mock:
            self.use_c_driver = True
            logger.info("MotorController using Native C Driver (libpca9685.so)!")
        else:
            self.use_c_driver = False
            logger.info("MotorController falling back to Python PCA9685 driver...")
            self.pca = PCA9685(bus_num=i2c_bus, address=i2c_address)
            self.pca.set_pwm_freq(60)

        self.left_pwm, self.left_in1, self.left_in2 = left_motor_channels
        self.right_pwm, self.right_in1, self.right_in2 = right_motor_channels

        self.stop()

    def set_left_speed(self, speed):
        """
        Set speed of left motor (-1.0 to 1.0).
        Positive speed = forward, Negative speed = reverse.
        """
        speed = max(-1.0, min(1.0, speed))
        duty = abs(speed)

        if speed > 0:
            # Forward: IN1 = High (4095), IN2 = Low (0)
            self.pca.set_pwm(self.left_in1, 4095, 0)
            self.pca.set_pwm(self.left_in2, 0, 0)
        elif speed < 0:
            # Reverse: IN1 = Low (0), IN2 = High (4095)
            self.pca.set_pwm(self.left_in1, 0, 0)
            self.pca.set_pwm(self.left_in2, 4095, 0)
        else:
            # Coast / Stop
            self.pca.set_pwm(self.left_in1, 0, 0)
            self.pca.set_pwm(self.left_in2, 0, 0)

        self.pca.set_duty_cycle(self.left_pwm, duty)

    def set_right_speed(self, speed):
        """
        Set speed of right motor (-1.0 to 1.0).
        Positive speed = forward, Negative speed = reverse.
        """
        speed = max(-1.0, min(1.0, speed))
        duty = abs(speed)

        if speed > 0:
            self.pca.set_pwm(self.right_in1, 4095, 0)
            self.pca.set_pwm(self.right_in2, 0, 0)
        elif speed < 0:
            self.pca.set_pwm(self.right_in1, 0, 0)
            self.pca.set_pwm(self.right_in2, 4095, 0)
        else:
            self.pca.set_pwm(self.right_in1, 0, 0)
            self.pca.set_pwm(self.right_in2, 0, 0)

        self.pca.set_duty_cycle(self.right_pwm, duty)

    def set_motors(self, left_speed, right_speed):
        """Set both motor speeds simultaneously."""
        if hasattr(self, 'use_c_driver') and self.use_c_driver:
            self.c_driver.set_motors(left_speed, right_speed)
        else:
            self.set_left_speed(left_speed)
            self.set_right_speed(right_speed)

    def drive_vector(self, linear_vel, steering):
        """
        Differential steering control.
        :param linear_vel: Forward/reverse throttle (-1.0 to 1.0)
        :param steering: Turn rate (-1.0 full left to +1.0 full right)
        """
        left = linear_vel + steering
        right = linear_vel - steering

        # Normalize if speeds exceed [-1.0, 1.0]
        max_val = max(abs(left), abs(right))
        if max_val > 1.0:
            left /= max_val
            right /= max_val

        self.set_motors(left, right)

    def stop(self):
        """Stop both motors."""
        self.set_motors(0.0, 0.0)
