/*
 * test_pca9685.c — Motor Driver Validation Test Suite
 *
 * Standalone C test binary for the PCA9685 I2C driver and TB6612FNG motor control.
 * Designed to run directly on the Jetson Nano to validate hardware connectivity
 * before integrating with the Python control stack.
 *
 * Build:   make test
 * Run:     sudo ./test_pca9685          (runs all tests)
 *          sudo ./test_pca9685 --scan   (scan I2C bus only, no motor movement)
 *          sudo ./test_pca9685 --motor  (skip to motor test directly)
 *
 * NOTE: Must run as root (sudo) because /dev/i2c-1 requires elevated permissions
 *       unless you've added your user to the i2c group:
 *       sudo usermod -aG i2c $USER && reboot
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/ioctl.h>
#include <math.h>
#include "pca9685.h"

#ifndef I2C_SLAVE
#define I2C_SLAVE 0x0703
#endif

/* ========================================================================= */
/*  ANSI Color Codes for Terminal Output                                     */
/* ========================================================================= */
#define GREEN   "\033[1;32m"
#define RED     "\033[1;31m"
#define YELLOW  "\033[1;33m"
#define CYAN    "\033[1;36m"
#define RESET   "\033[0m"

#define PASS(msg)  printf("  " GREEN "[PASS]" RESET " %s\n", msg)
#define FAIL(msg)  printf("  " RED   "[FAIL]" RESET " %s\n", msg)
#define INFO(msg)  printf("  " CYAN  "[INFO]" RESET " %s\n", msg)
#define WARN(msg)  printf("  " YELLOW "[WARN]" RESET " %s\n", msg)

/* Test counters */
static int tests_passed = 0;
static int tests_failed = 0;

#define ASSERT_TRUE(cond, msg) do {     \
    if (cond) {                         \
        PASS(msg);                      \
        tests_passed++;                 \
    } else {                            \
        FAIL(msg);                      \
        tests_failed++;                 \
    }                                   \
} while(0)


/* ========================================================================= */
/*  TEST 1: I2C Bus Scan                                                     */
/*  Scans all 128 addresses on /dev/i2c-1 to find responsive devices.        */
/*  Expected: PCA9685 responds at 0x60 (or 0x40 on some boards).             */
/* ========================================================================= */
void test_i2c_bus_scan(const char* device) {
    printf("\n" CYAN "═══ TEST 1: I2C Bus Scan ═══" RESET "\n");
    printf("  Scanning %s for responsive devices...\n\n", device);

    int fd = open(device, O_RDWR);
    if (fd < 0) {
        FAIL("Cannot open I2C device. Are you running as root (sudo)?");
        tests_failed++;
        return;
    }

    int found_count = 0;
    int found_pca9685 = 0;

    printf("       0  1  2  3  4  5  6  7  8  9  a  b  c  d  e  f\n");
    for (int row = 0; row < 8; row++) {
        printf("  %02x:", row * 16);
        for (int col = 0; col < 16; col++) {
            int addr = row * 16 + col;

            /* Skip reserved addresses (0x00-0x02, 0x78-0x7F) */
            if (addr < 0x03 || addr > 0x77) {
                printf("   ");
                continue;
            }

            if (ioctl(fd, I2C_SLAVE, addr) < 0) {
                printf("   ");
                continue;
            }

            /* Try reading byte — if the device ACKs, it exists */
            unsigned char reg = 0x00;
            if (write(fd, &reg, 1) == 1) {
                unsigned char val;
                if (read(fd, &val, 1) == 1) {
                    printf(" %02x", addr);
                    found_count++;
                    if (addr == 0x60 || addr == 0x40) {
                        found_pca9685 = 1;
                    }
                } else {
                    printf(" --");
                }
            } else {
                printf(" --");
            }
        }
        printf("\n");
    }

    printf("\n");
    ASSERT_TRUE(found_count > 0, "At least one I2C device found on bus");
    ASSERT_TRUE(found_pca9685, "PCA9685 found at expected address (0x60 or 0x40)");
    printf("  Found %d device(s) total.\n", found_count);

    close(fd);
}


/* ========================================================================= */
/*  TEST 2: PCA9685 Initialization                                           */
/*  Verifies we can open the I2C bus, configure the PCA9685, and read back   */
/*  register values to confirm the chip is responding correctly.              */
/* ========================================================================= */
void test_pca9685_init(const char* device, int addr) {
    printf("\n" CYAN "═══ TEST 2: PCA9685 Initialization ═══" RESET "\n");

    int fd = pca9685_init(device, addr);
    ASSERT_TRUE(fd >= 0, "pca9685_init() returned valid file descriptor");

    if (fd < 0) {
        FAIL("Cannot continue — init failed. Check wiring and I2C address.");
        return;
    }

    /* Read back MODE1 register — should be 0x01 (ALLCALL) after our reset */
    ioctl(fd, I2C_SLAVE, addr);
    unsigned char reg = PCA9685_MODE1;
    write(fd, &reg, 1);
    unsigned char mode1_val;
    read(fd, &mode1_val, 1);

    printf("  MODE1 register value: 0x%02X\n", mode1_val);
    /*
     * After reset, MODE1 should have ALLCALL (bit 0) set.
     * The RESTART bit (bit 7) and SLEEP bit (bit 4) should be clear.
     * We check bit 0 is set and bit 4 (SLEEP) is clear.
     */
    ASSERT_TRUE((mode1_val & 0x01) == 0x01, "MODE1 ALLCALL bit is set (bit 0 = 1)");
    ASSERT_TRUE((mode1_val & 0x10) == 0x00, "MODE1 SLEEP bit is clear (bit 4 = 0, chip is awake)");

    /* Read back MODE2 register — should be 0x04 (OUTDRV) */
    reg = PCA9685_MODE2;
    write(fd, &reg, 1);
    unsigned char mode2_val;
    read(fd, &mode2_val, 1);

    printf("  MODE2 register value: 0x%02X\n", mode2_val);
    ASSERT_TRUE((mode2_val & 0x04) == 0x04, "MODE2 OUTDRV bit is set (totem-pole output)");

    pca9685_close(fd);
    PASS("PCA9685 closed cleanly");
}


/* ========================================================================= */
/*  TEST 3: PWM Frequency Configuration                                     */
/*  Sets the prescaler to 60Hz and reads it back to verify.                  */
/*  Expected prescale value for 60Hz: round(25MHz / (4096 * 60)) - 1 = 101  */
/* ========================================================================= */
void test_pwm_frequency(const char* device, int addr) {
    printf("\n" CYAN "═══ TEST 3: PWM Frequency (Prescaler) ═══" RESET "\n");

    int fd = pca9685_init(device, addr);
    if (fd < 0) { FAIL("Init failed"); return; }

    /* Set to 60 Hz */
    int ret = pca9685_set_pwm_freq(fd, addr, 60.0f);
    ASSERT_TRUE(ret == 0, "pca9685_set_pwm_freq(60Hz) returned success");

    /* Read back the prescale register to verify */
    ioctl(fd, I2C_SLAVE, addr);

    /* Must put chip to sleep to read prescale (datasheet requirement) */
    unsigned char reg = PCA9685_MODE1;
    write(fd, &reg, 1);
    unsigned char mode1_val;
    read(fd, &mode1_val, 1);

    unsigned char sleep_mode = (mode1_val & 0x7F) | 0x10;
    unsigned char buf[2] = {PCA9685_MODE1, sleep_mode};
    write(fd, buf, 2);
    usleep(5000);

    reg = PCA9685_PRESCALE;
    write(fd, &reg, 1);
    unsigned char prescale_val;
    read(fd, &prescale_val, 1);

    /* Wake chip back up */
    buf[0] = PCA9685_MODE1;
    buf[1] = mode1_val;
    write(fd, buf, 2);
    usleep(5000);
    buf[1] = mode1_val | 0x80;
    write(fd, buf, 2);

    float expected_prescale = roundf(25000000.0f / (4096.0f * 60.0f) - 1.0f);
    printf("  Read prescale register: %d (expected: %.0f for 60Hz)\n",
           prescale_val, expected_prescale);

    /* Allow +/- 1 tolerance for rounding */
    int diff = abs((int)prescale_val - (int)expected_prescale);
    ASSERT_TRUE(diff <= 1, "Prescale value matches expected (within +/-1 tolerance)");

    pca9685_close(fd);
}


/* ========================================================================= */
/*  TEST 4: Individual PWM Channel Output                                    */
/*  Sets a known duty cycle on channel 0, reads it back, then clears it.     */
/* ========================================================================= */
void test_pwm_channel_readback(const char* device, int addr) {
    printf("\n" CYAN "═══ TEST 4: PWM Channel Register Read-Back ═══" RESET "\n");

    int fd = pca9685_init(device, addr);
    if (fd < 0) { FAIL("Init failed"); return; }
    pca9685_set_pwm_freq(fd, addr, 60.0f);

    /* Set channel 0: ON at tick 0, OFF at tick 2048 (50% duty cycle) */
    int test_channel = 0;
    int test_on = 0;
    int test_off = 2048;

    pca9685_set_pwm(fd, addr, test_channel, test_on, test_off);

    /* Read back the 4 registers for channel 0 */
    ioctl(fd, I2C_SLAVE, addr);

    unsigned char reg, val;
    int readback_on, readback_off;

    /* ON_L */
    reg = LED0_ON_L + 4 * test_channel;
    write(fd, &reg, 1); read(fd, &val, 1);
    readback_on = val;

    /* ON_H */
    reg = LED0_ON_H + 4 * test_channel;
    write(fd, &reg, 1); read(fd, &val, 1);
    readback_on |= (val << 8);

    /* OFF_L */
    reg = LED0_OFF_L + 4 * test_channel;
    write(fd, &reg, 1); read(fd, &val, 1);
    readback_off = val;

    /* OFF_H */
    reg = LED0_OFF_H + 4 * test_channel;
    write(fd, &reg, 1); read(fd, &val, 1);
    readback_off |= (val << 8);

    printf("  Wrote: ON=%d, OFF=%d\n", test_on, test_off);
    printf("  Read:  ON=%d, OFF=%d\n", readback_on, readback_off);

    ASSERT_TRUE(readback_on == test_on, "ON tick read-back matches written value");
    ASSERT_TRUE(readback_off == test_off, "OFF tick read-back matches written value");

    /* Clean up — zero the channel */
    pca9685_set_pwm(fd, addr, test_channel, 0, 0);
    pca9685_close(fd);
}


/* ========================================================================= */
/*  TEST 5: Motor Direction & Speed (INTERACTIVE — MOTORS WILL SPIN!)        */
/*  Runs each motor forward, reverse, then stop. User visually confirms.     */
/*  Skipped unless --motor flag is passed.                                   */
/* ========================================================================= */
void test_motor_movement(const char* device, int addr) {
    printf("\n" CYAN "═══ TEST 5: Motor Movement (INTERACTIVE) ═══" RESET "\n");
    WARN("⚠  MOTORS WILL SPIN. Place JetBot on a raised surface (wheels off ground)!");
    printf("  Press ENTER to continue, or Ctrl+C to abort...\n");
    getchar();

    int fd = pca9685_init(device, addr);
    if (fd < 0) { FAIL("Init failed"); return; }
    pca9685_set_pwm_freq(fd, addr, 60.0f);

    /* --- Left Motor Forward 30% for 1.5 seconds --- */
    INFO("Left motor FORWARD at 30%...");
    pca9685_set_motor_speeds(fd, addr, 0.3f, 0.0f);
    sleep(1);
    pca9685_set_motor_speeds(fd, addr, 0.0f, 0.0f);
    usleep(500000);

    /* --- Left Motor Reverse 30% for 1.5 seconds --- */
    INFO("Left motor REVERSE at 30%...");
    pca9685_set_motor_speeds(fd, addr, -0.3f, 0.0f);
    sleep(1);
    pca9685_set_motor_speeds(fd, addr, 0.0f, 0.0f);
    usleep(500000);

    /* --- Right Motor Forward 30% for 1.5 seconds --- */
    INFO("Right motor FORWARD at 30%...");
    pca9685_set_motor_speeds(fd, addr, 0.0f, 0.3f);
    sleep(1);
    pca9685_set_motor_speeds(fd, addr, 0.0f, 0.0f);
    usleep(500000);

    /* --- Right Motor Reverse 30% for 1.5 seconds --- */
    INFO("Right motor REVERSE at 30%...");
    pca9685_set_motor_speeds(fd, addr, 0.0f, -0.3f);
    sleep(1);
    pca9685_set_motor_speeds(fd, addr, 0.0f, 0.0f);
    usleep(500000);

    /* --- Both Forward (straight) --- */
    INFO("Both motors FORWARD at 25% (straight drive)...");
    pca9685_set_motor_speeds(fd, addr, 0.25f, 0.25f);
    sleep(1);
    pca9685_set_motor_speeds(fd, addr, 0.0f, 0.0f);
    usleep(500000);

    /* --- Spin in place (left forward, right reverse) --- */
    INFO("Spin in place (left fwd, right rev at 25%)...");
    pca9685_set_motor_speeds(fd, addr, 0.25f, -0.25f);
    sleep(1);

    /* --- Full stop --- */
    pca9685_set_motor_speeds(fd, addr, 0.0f, 0.0f);
    INFO("All motors stopped.");

    printf("\n  Did the motors behave as described above?\n");
    printf("  - Left motor spun forward, then reverse\n");
    printf("  - Right motor spun forward, then reverse\n");
    printf("  - Both drove forward together\n");
    printf("  - Robot spun in place\n");
    PASS("Motor movement test sequence completed (verify visually)");

    pca9685_close(fd);
}


/* ========================================================================= */
/*  TEST 6: Boundary / Edge Case Validation                                  */
/*  Tests invalid inputs, out-of-range channels, and clamping behavior.      */
/* ========================================================================= */
void test_edge_cases(const char* device, int addr) {
    printf("\n" CYAN "═══ TEST 6: Edge Cases & Boundary Validation ═══" RESET "\n");

    int fd = pca9685_init(device, addr);
    if (fd < 0) { FAIL("Init failed"); return; }
    pca9685_set_pwm_freq(fd, addr, 60.0f);

    /* Invalid channel should return -1 */
    int ret;
    ret = pca9685_set_pwm(fd, addr, -1, 0, 0);
    ASSERT_TRUE(ret == -1, "set_pwm with channel -1 returns error (-1)");

    ret = pca9685_set_pwm(fd, addr, 16, 0, 0);
    ASSERT_TRUE(ret == -1, "set_pwm with channel 16 returns error (-1)");

    /* Valid boundary channels */
    ret = pca9685_set_pwm(fd, addr, 0, 0, 0);
    ASSERT_TRUE(ret == 0, "set_pwm with channel 0 succeeds");

    ret = pca9685_set_pwm(fd, addr, 15, 0, 0);
    ASSERT_TRUE(ret == 0, "set_pwm with channel 15 succeeds");

    /* Motor speed clamping: speed > 1.0 should clamp internally */
    ret = pca9685_set_motor_speeds(fd, addr, 1.5f, -1.5f);
    ASSERT_TRUE(ret == 0, "set_motor_speeds with out-of-range speeds doesn't crash");

    /* Zero speed — motors should coast */
    ret = pca9685_set_motor_speeds(fd, addr, 0.0f, 0.0f);
    ASSERT_TRUE(ret == 0, "set_motor_speeds(0, 0) succeeds (coast stop)");

    /* Invalid fd */
    ret = pca9685_set_pwm(-1, addr, 0, 0, 0);
    ASSERT_TRUE(ret == -1, "set_pwm with invalid fd returns error (-1)");

    ret = pca9685_set_motor_speeds(-1, addr, 0.5f, 0.5f);
    ASSERT_TRUE(ret == -1, "set_motor_speeds with invalid fd returns error (-1)");

    pca9685_set_motor_speeds(fd, addr, 0.0f, 0.0f);
    pca9685_close(fd);
}


/* ========================================================================= */
/*  MAIN — Test Runner                                                       */
/* ========================================================================= */
int main(int argc, char* argv[]) {
    const char* i2c_device = "/dev/i2c-1";
    int i2c_addr = 0x60;

    int run_scan_only = 0;
    int run_motor_only = 0;

    /* Parse command line flags */
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--scan") == 0) {
            run_scan_only = 1;
        } else if (strcmp(argv[i], "--motor") == 0) {
            run_motor_only = 1;
        } else if (strcmp(argv[i], "--addr") == 0 && i + 1 < argc) {
            i2c_addr = (int)strtol(argv[++i], NULL, 16);
        } else if (strcmp(argv[i], "--help") == 0) {
            printf("Usage: %s [OPTIONS]\n", argv[0]);
            printf("  --scan         Scan I2C bus only (no motor movement)\n");
            printf("  --motor        Skip to interactive motor test\n");
            printf("  --addr 0xNN    Use different I2C address (default: 0x60)\n");
            printf("  --help         Show this help\n");
            return 0;
        }
    }

    printf(CYAN "\n╔══════════════════════════════════════════════════════════╗\n");
    printf("║   PCA9685 + TB6612FNG Motor Driver Test Suite            ║\n");
    printf("║   Target: Jetson Nano — /dev/i2c-1 @ 0x%02X               ║\n", i2c_addr);
    printf("╚══════════════════════════════════════════════════════════╝\n" RESET);

    if (run_scan_only) {
        test_i2c_bus_scan(i2c_device);
    } else if (run_motor_only) {
        /* Always verify I2C connectivity before spinning motors */
        test_i2c_bus_scan(i2c_device);
        test_pca9685_init(i2c_device, i2c_addr);
        test_pwm_frequency(i2c_device, i2c_addr);
        test_motor_movement(i2c_device, i2c_addr);
    } else {
        /* Run full test suite in order */
        test_i2c_bus_scan(i2c_device);
        test_pca9685_init(i2c_device, i2c_addr);
        test_pwm_frequency(i2c_device, i2c_addr);
        test_pwm_channel_readback(i2c_device, i2c_addr);
        test_edge_cases(i2c_device, i2c_addr);

        printf("\n" YELLOW "═══ Skipping Motor Movement Test (use --motor flag) ═══" RESET "\n");
        printf("  Run with:  sudo ./test_pca9685 --motor\n");
    }

    /* Summary */
    printf("\n" CYAN "════════════════════════════════════════════\n");
    printf("  TEST SUMMARY\n");
    printf("════════════════════════════════════════════" RESET "\n");
    printf("  " GREEN "Passed: %d" RESET "\n", tests_passed);
    printf("  " RED   "Failed: %d" RESET "\n", tests_failed);
    printf("  Total:  %d\n", tests_passed + tests_failed);
    printf(CYAN "════════════════════════════════════════════\n" RESET);

    return tests_failed > 0 ? 1 : 0;
}
