# Low-Level Motor Driver Architecture: PCA9685 & TB6612FNG Deep-Dive

> **Target Hardware:** Waveshare JetBot Motor HAT / Adafruit Motor HAT  
> **Host Processor:** NVIDIA Jetson Nano (ARM64, Linux `/dev/i2c-1`)  
> **Primary Components:** NXP PCA9685 16-channel 12-bit PWM controller + Toshiba TB6612FNG Dual H-Bridge Driver  
> **Official Datasheet:** [NXP PCA9685 Product Datasheet (PDF)](https://www.nxp.com/docs/en/data-sheet/PCA9685.pdf)

---

## Table of Contents
1. [System Architecture Overview](#1-system-architecture-overview)
2. [I2C Protocol & Addressing](#2-i2c-protocol--addressing)
3. [PCA9685 Internal Registers](#3-pca9685-internal-registers)
4. [Prescaler Math & PWM Frequency Logic](#4-prescaler-math--pwm-frequency-logic)
5. [12-Bit PWM Counter & "Bit 12" Control Logic](#5-12-bit-pwm-counter--bit-12-control-logic)
6. [TB6612FNG Dual H-Bridge & Pin Mapping](#6-tb6612fng-dual-h-bridge--pin-mapping)
7. [Differential Drive Kinematics Math](#7-differential-drive-kinematics-math)
8. [C Driver Implementation (`pca9685.c`) Step-by-Step](#8-c-driver-implementation-pca9685c-step-by-step)
9. [C-to-Python Integration via `ctypes`](#9-c-to-python-integration-via-ctypes)

---

<a id="1-system-architecture-overview"></a>
## 1. System Architecture Overview

The Jetson Nano CPU does not directly output high-current motor voltages. Instead, it uses a two-stage control topology:

```
┌─────────────────┐       I2C Bus        ┌──────────────────┐    GPIO Control    ┌──────────────────┐   Motor Power  ┌──────────────┐
│   Jetson Nano   │ ───────────────────> │  PCA9685 PWM IC  │ ─────────────────> │ TB6612FNG Driver │ -------------> │  DC Motors   │
│ (Linux /dev/i2c)│  SCL/SDA @ 0x60      │ (16-ch 12-bit)   │  PWM + DIR Signals │ (Dual H-Bridge)  │   (V+ / V-)    │ (Left/Right) │
└─────────────────┘                      └──────────────────┘                    └──────────────────┘                └──────────────┘
```

1. **Jetson Nano:** Sends binary I2C bytes over the `/dev/i2c-1` bus (GPIO pins 3 and 5) to set register values.
2. **PCA9685 IC:** Translates register values into precise square-wave PWM timing signals (0V to 3.3V).
3. **TB6612FNG Driver:** Uses the low-power PWM signals from the PCA9685 to gate high-current battery power (6V–12V) directly into the DC motors.

---

<a id="2-i2c-protocol--addressing"></a>
## 2. I2C Protocol & Addressing

### Bus Topology
- **Device Node:** `/dev/i2c-1` (Jetson Nano expansion header I2C Bus 1).
- **Target Address:** `0x60` (7-bit address: `0b1100000`).
- **All-Call Address:** `0x70` (7-bit address used to broadcast commands to all PCA9685 chips simultaneously).

### Data Frame Byte Sequence
Every write operation to the PCA9685 follows the standard 2-byte I2C register write sequence:

```
START ──> [Address + Write (0xC0)] ──> [Register Pointer Byte] ──> [Data Byte] ──> STOP
```

To write 16-bit values (like ON/OFF counter ticks), two sequential byte writes are required:
1. Low byte (`_L`)
2. High byte (`_H`)

---

<a id="3-pca9685-internal-registers"></a>
## 3. PCA9685 Internal Registers

| Register Name | Hex Address | Binary Mask / Bits | Purpose / Logic |
|---|---|---|---|
| **`MODE1`** | `0x00` | `[RESTART \| EXTCLK \| AI \| SLEEP \| SUB1 \| SUB2 \| SUB3 \| ALLCALL]` | Control register 1 (sleep, restart, clock selection). |
| **`MODE2`** | `0x01` | `[0 \| 0 \| 0 \| INVRT \| OUTDRV \| OUTNE1 \| OUTNE0]` | Control register 2 (output structure: totem-pole vs open-drain). |
| **`PRESCALE`** | `0xFE` | `[7:0]` | Sets PWM frequency prescaler (read/writeable ONLY during sleep mode). |
| **`LED0_ON_L`** | `0x06` | `[7:0]` | Channel 0 ON tick low byte. |
| **`LED0_ON_H`** | `0x07` | `[11:8]` + `[12]` (FULL ON) | Channel 0 ON tick high byte + Bit 12 override. |
| **`LED0_OFF_L`** | `0x08` | `[7:0]` | Channel 0 OFF tick low byte. |
| **`LED0_OFF_H`** | `0x09` | `[11:8]` + `[12]` (FULL OFF) | Channel 0 OFF tick high byte + Bit 12 override. |

*Note: Subsequent channels are offset by 4 bytes. Channel `n` registers are at `0x06 + (4 * n)`.*

---

<a id="4-prescaler-math--pwm-frequency-logic"></a>
## 4. Prescaler Math & PWM Frequency Logic

The PCA9685 contains an internal **25 MHz master clock ($f_{osc} = 25,000,000\text{ Hz}$)**. The 12-bit PWM counter updates every clock tick up to 4096 counts ($2^{12}$).

The formula to calculate the 8-bit `PRESCALE` register value for a target frequency ($f_{target}$) is:

$$\text{prescale} = \text{round}\left( \frac{f_{osc}}{4096 \times f_{target}} \right) - 1$$

### Example Calculation for 60 Hz Motor Control:
$$\text{prescale} = \text{round}\left( \frac{25,000,000}{4096 \times 60} \right) - 1 = \text{round}\left( \frac{25,000,000}{245,760} \right) - 1 = \text{round}(101.725) - 1 = 102 - 1 = 101 \text{ (0x65)}$$

### Register Transition Sequence for Frequency Change:
1. Read current `MODE1` byte (`0x00`).
2. Set `SLEEP` bit (bit 4 = 1) to pause the oscillator: `newmode = (oldmode & 0x7F) | 0x10`.
3. Write `newmode` to `MODE1` (`0x00`).
4. Write calculated prescale byte (`101`) to `PRESCALE` (`0xFE`).
5. Restore `oldmode` to wake the oscillator (bit 4 = 0).
6. Wait 5 milliseconds (`usleep(5000)`) for clock stabilization.
7. Set `RESTART` bit (bit 7 = 1) to restart the PWM counter logic.

---

<a id="5-12-bit-pwm-counter--bit-12-control-logic"></a>
## 5. 12-Bit PWM Counter & "Bit 12" Control Logic

Each channel has a 12-bit counter running from tick `0` to tick `4095`.

```
Tick: 0                        OFF_TICK                 4095
      │───────────────────────────│───────────────────────│
Signal: ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓                        (Square Wave Duty Cycle)
        <────── HIGH (on) ────────><──── LOW (off) ───────>
```

- **Variable PWM Duty Cycle (e.g., 50% Speed):**  
  `ON = 0`, `OFF = 2047`  
  *Signal turns HIGH at tick 0 and goes LOW at tick 2047.*

- **Full HIGH (100% On / Logic 1 for direction pins):**  
  Set **Bit 12 of `ON_H`** (`0x10` in high byte). `ON = 4096`, `OFF = 0`.  
  *Forces output pin continuously HIGH regardless of counter.*

- **Full LOW (0% Off / Logic 0 for direction pins):**  
  Set **Bit 12 of `OFF_H`** (`0x10` in high byte). `ON = 0`, `OFF = 4096`.  
  *Forces output pin continuously LOW regardless of counter.*

---

<a id="6-tb6612fng-dual-h-bridge--pin-mapping"></a>
## 6. TB6612FNG Dual H-Bridge & Pin Mapping

The TB6612FNG uses two direction inputs (`IN1`, `IN2`) and one speed input (`PWM`) per motor:

### Direction Control Truth Table:
| `IN1` Pin State | `IN2` Pin State | H-Bridge Output State | Motor Direction |
|---|---|---|---|
| **LOW (`0`)** | **HIGH (`1`)** | OUT1 < OUT2 | **FORWARD (Camera direction)** |
| **HIGH (`1`)** | **LOW (`0`)** | OUT1 > OUT2 | **REVERSE** |
| **LOW (`0`)** | **LOW (`0`)** | High Impedance (OFF) | **Coast Stop** |

### Waveshare / Adafruit Motor HAT Channel Assignments:

```
Left Motor (Motor 1):
  ├── Channel 8  ──> PWM  (Speed Duty Cycle 0-4095)
  ├── Channel 10 ──> IN1  (Direction Pin 1)
  └── Channel 9  ──> IN2  (Direction Pin 2)

Right Motor (Motor 2):
  ├── Channel 13 ──> PWM  (Speed Duty Cycle 0-4095)
  ├── Channel 12 ──> IN1  (Direction Pin 1)
  └── Channel 11 ──> IN2  (Direction Pin 2)
```

---

<a id="7-differential-drive-kinematics-math"></a>
## 7. Differential Drive Kinematics Math

The high-level `MotorController` accepts normalized linear velocity ($v \in [-1.0, 1.0]$) and steering angle ($\omega \in [-1.0, 1.0]$).

### Differential Kinematic Equations:
$$v_{left} = v + \omega$$
$$v_{right} = v - \omega$$

### Normalization & Clamping:
To prevent motor saturation when $|v| + |\omega| > 1.0$:

$$\text{max\_val} = \max(1.0, |v_{left}|, |v_{right}|)$$
$$v_{left\_norm} = \frac{v_{left}}{\text{max\_val}}, \quad v_{right\_norm} = \frac{v_{right}}{\text{max\_val}}$$

---

<a id="8-c-driver-implementation-pca9685c-step-by-step"></a>
## 8. C Driver Implementation (`pca9685.c`) Step-by-Step

The C driver interfaces directly with POSIX `/dev/i2c-1` kernel calls:

1. **Initialization (`pca9685_init`):**
   - Calls `open("/dev/i2c-1", O_RDWR)` to open the Linux file descriptor.
   - Executes `ioctl(fd, I2C_SLAVE, 0x60)` to bind the file descriptor to address `0x60`.
   - Writes `MODE1` (`0x00` -> `0x01`) and `MODE2` (`0x01` -> `0x04`).

2. **Setting Motor Speeds (`pca9685_set_motor_speeds`):**
   - Calculates absolute duty cycle count: `pwm = (int)(abs(speed) * 4095)`.
   - Evaluates sign of `left_speed`:
     - If `> 0`: Writes `IN1(ch10) = LOW`, `IN2(ch9) = HIGH`, `PWM(ch8) = l_pwm`.
     - If `< 0`: Writes `IN1(ch10) = HIGH`, `IN2(ch9) = LOW`, `PWM(ch8) = l_pwm`.
     - If `== 0`: Writes `IN1 = LOW`, `IN2 = LOW`, `PWM = 0`.

---

<a id="9-c-to-python-integration-via-ctypes"></a>
## 9. C-to-Python Integration via `ctypes`

To combine C performance with Python usability:

1. **Compilation:** `gcc -shared -fPIC -O2 -o libpca9685.so pca9685.c` compiles C code into a shared binary object.
2. **Python Foreign Function Interface (`ctypes`):**
   - `lib = ctypes.CDLL("./libpca9685.so")` loads the compiled library into Python runtime memory.
   - Functions are registered with explicit C parameter types (`argtypes`) and return types (`restype`).
   - High-level Python scripts call `lib.pca9685_set_motor_speeds(...)` directly, bypassing Python interpreter overhead for low-level I2C transactions.
