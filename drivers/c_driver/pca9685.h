#ifndef PCA9685_H
#define PCA9685_H

#ifdef __cplusplus
extern "C" {
#endif

// PCA9685 Registers
#define PCA9685_MODE1       0x00
#define PCA9685_MODE2       0x01
#define PCA9685_PRESCALE    0xFE
#define LED0_ON_L           0x06
#define LED0_ON_H           0x07
#define LED0_OFF_L          0x08
#define LED0_OFF_H          0x09

// Functions exposed to Python via ctypes
int pca9685_init(const char* i2c_device, int addr);
int pca9685_set_pwm_freq(int fd, int addr, float freq_hz);
int pca9685_set_pwm(int fd, int addr, int channel, int on, int off);
int pca9685_set_motor_speeds(int fd, int addr, float left_speed, float right_speed);
void pca9685_close(int fd);

#ifdef __cplusplus
}
#endif

#endif // PCA9685_H
