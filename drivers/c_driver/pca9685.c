#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <fcntl.h>
#include <math.h>
#include <sys/ioctl.h>
#include "pca9685.h"

// Linux I2C definition fallback if i2c-dev header is missing
#ifndef I2C_SLAVE
#define I2C_SLAVE 0x0703
#endif

static int i2c_write_byte(int fd, unsigned char reg, unsigned char data) {
    unsigned char buf[2] = {reg, data};
    if (write(fd, buf, 2) != 2) {
        return -1;
    }
    return 0;
}

static int i2c_read_byte(int fd, unsigned char reg) {
    if (write(fd, &reg, 1) != 1) return -1;
    unsigned char val;
    if (read(fd, &val, 1) != 1) return -1;
    return val;
}

int pca9685_init(const char* i2c_device, int addr) {
    int fd = open(i2c_device, O_RDWR);
    if (fd < 0) {
        perror("Failed to open I2C device bus");
        return -1;
    }
    if (ioctl(fd, I2C_SLAVE, addr) < 0) {
        perror("Failed to acquire bus access / talk to slave");
        close(fd);
        return -1;
    }
    
    // Reset PCA9685
    i2c_write_byte(fd, PCA9685_MODE1, 0x01); // ALLCALL
    i2c_write_byte(fd, PCA9685_MODE2, 0x04); // OUTDRV
    usleep(5000);
    return fd;
}

int pca9685_set_pwm_freq(int fd, int addr, float freq_hz) {
    if (fd < 0) return -1;
    ioctl(fd, I2C_SLAVE, addr);

    float prescaleval = 25000000.0f / 4096.0f / freq_hz - 1.0f;
    unsigned char prescale = (unsigned char)floor(prescaleval + 0.5f);

    int oldmode = i2c_read_byte(fd, PCA9685_MODE1);
    int newmode = (oldmode & 0x7F) | 0x10; // Sleep mode to write prescale

    i2c_write_byte(fd, PCA9685_MODE1, newmode);
    i2c_write_byte(fd, PCA9685_PRESCALE, prescale);
    i2c_write_byte(fd, PCA9685_MODE1, oldmode);
    usleep(5000);
    i2c_write_byte(fd, PCA9685_MODE1, oldmode | 0x80); // Restart
    return 0;
}

int pca9685_set_pwm(int fd, int addr, int channel, int on, int off) {
    if (fd < 0 || channel < 0 || channel > 15) return -1;
    ioctl(fd, I2C_SLAVE, addr);

    i2c_write_byte(fd, LED0_ON_L + 4 * channel, on & 0xFF);
    i2c_write_byte(fd, LED0_ON_H + 4 * channel, (on >> 8) & 0xFF);
    i2c_write_byte(fd, LED0_OFF_L + 4 * channel, off & 0xFF);
    i2c_write_byte(fd, LED0_OFF_H + 4 * channel, (off >> 8) & 0xFF);
    return 0;
}

int pca9685_set_motor_speeds(int fd, int addr, float left_speed, float right_speed) {
    if (fd < 0) return -1;

    /*
     * Waveshare / Adafruit Motor HAT — PCA9685 Channel Mapping:
     *   Left Motor (Motor 1):  PWM=ch8,  IN1=ch10, IN2=ch9
     *   Right Motor (Motor 2): PWM=ch13, IN1=ch12, IN2=ch11
     *
     * TB6612FNG direction truth table:
     *   IN1=HIGH, IN2=LOW  → Forward
     *   IN1=LOW,  IN2=HIGH → Reverse
     *   IN1=LOW,  IN2=LOW  → Coast (stop)
     *
     * PCA9685 "fully on" trick:
     *   To drive a pin fully HIGH:  set_pwm(ch, 4096, 0)  — bit 12 of ON = "always on"
     *   To drive a pin fully LOW:   set_pwm(ch, 0, 4096)  — bit 12 of OFF = "always off"
     *   For duty cycle PWM:         set_pwm(ch, 0, duty)   — ON at tick 0, OFF at tick 'duty'
     */

    // --- Left Motor (Motor 1: PWM 8, IN1 10, IN2 9) ---
    float l_duty = fabsf(left_speed);
    if (l_duty > 1.0f) l_duty = 1.0f;
    int l_pwm = (int)(l_duty * 4095.0f);

    if (left_speed > 0.001f) {
        pca9685_set_pwm(fd, addr, 10, 0, 4096);  // IN1 = LOW
        pca9685_set_pwm(fd, addr, 9, 4096, 0);   // IN2 = HIGH (camera forward)
    } else if (left_speed < -0.001f) {
        pca9685_set_pwm(fd, addr, 10, 4096, 0);  // IN1 = HIGH
        pca9685_set_pwm(fd, addr, 9, 0, 4096);   // IN2 = LOW
    } else {
        pca9685_set_pwm(fd, addr, 10, 0, 4096);
        pca9685_set_pwm(fd, addr, 9, 0, 4096);
        l_pwm = 0;
    }
    pca9685_set_pwm(fd, addr, 8, 0, l_pwm);      // PWM duty cycle

    // --- Right Motor (Motor 2: PWM 13, IN1 12, IN2 11) ---
    float r_duty = fabsf(right_speed);
    if (r_duty > 1.0f) r_duty = 1.0f;
    int r_pwm = (int)(r_duty * 4095.0f);

    if (right_speed > 0.001f) {
        pca9685_set_pwm(fd, addr, 12, 0, 4096);  // IN1 = LOW
        pca9685_set_pwm(fd, addr, 11, 4096, 0);  // IN2 = HIGH (camera forward)
    } else if (right_speed < -0.001f) {
        pca9685_set_pwm(fd, addr, 12, 4096, 0);  // IN1 = HIGH
        pca9685_set_pwm(fd, addr, 11, 0, 4096);  // IN2 = LOW
    } else {
        pca9685_set_pwm(fd, addr, 12, 0, 4096);
        pca9685_set_pwm(fd, addr, 11, 0, 4096);
        r_pwm = 0;
    }
    pca9685_set_pwm(fd, addr, 13, 0, r_pwm);     // PWM duty cycle

    return 0;
}

void pca9685_close(int fd) {
    if (fd >= 0) {
        close(fd);
    }
}
