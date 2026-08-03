import os
import ctypes
import logging

logger = logging.getLogger("C_PCA9685")

class CPCA9685Driver:
    """
    Python wrapper over native compiled C shared library libpca9685.so via ctypes.
    Provides ultra low-latency I2C register access on Linux/Jetson.
    """
    def __init__(self, i2c_device="/dev/i2c-1", address=0x60):
        self.i2c_device = i2c_device.encode('utf-8')
        self.address = address
        self.fd = -1
        self.lib = None
        self.is_mock = False

        so_path = os.path.join(os.path.dirname(__file__), "libpca9685.so")
        if not os.path.exists(so_path):
            logger.warning(f"Shared library {so_path} not found. Build it using 'make' inside drivers/c_driver/. Running in MOCK mode.")
            self.is_mock = True
            return

        try:
            self.lib = ctypes.CDLL(so_path)
            
            # Function signatures
            self.lib.pca9685_init.argtypes = [ctypes.c_char_p, ctypes.c_int]
            self.lib.pca9685_init.restype = ctypes.c_int

            self.lib.pca9685_set_pwm_freq.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_float]
            self.lib.pca9685_set_pwm_freq.restype = ctypes.c_int

            self.lib.pca9685_set_motor_speeds.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_float, ctypes.c_float]
            self.lib.pca9685_set_motor_speeds.restype = ctypes.c_int

            self.lib.pca9685_close.argtypes = [ctypes.c_int]
            self.lib.pca9685_close.restype = None

            self.fd = self.lib.pca9685_init(self.i2c_device, self.address)
            if self.fd < 0:
                logger.warning("C pca9685_init failed. Running in MOCK mode.")
                self.is_mock = True
            else:
                self.lib.pca9685_set_pwm_freq(self.fd, self.address, ctypes.c_float(60.0))
                logger.info("Successfully initialized native C PCA9685 driver!")

        except Exception as e:
            logger.warning(f"Failed to load C library: {e}. Running in MOCK mode.")
            self.is_mock = True

    def set_motors(self, left_speed, right_speed):
        if self.is_mock or not self.lib or self.fd < 0:
            return
        self.lib.pca9685_set_motor_speeds(self.fd, self.address, ctypes.c_float(left_speed), ctypes.c_float(right_speed))

    def close(self):
        if self.lib and self.fd >= 0:
            self.set_motors(0.0, 0.0)
            self.lib.pca9685_close(self.fd)
            self.fd = -1
