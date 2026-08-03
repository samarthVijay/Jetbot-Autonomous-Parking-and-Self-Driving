# JetBot Autonomous Parking & Self-Driving System

[![NVIDIA Jetson Nano](https://img.shields.io/badge/Hardware-NVIDIA%20Jetson%20Nano-green.svg)](https://developer.nvidia.com/embedded/jetson-nano-developer-kit)
[![Python 3.6+](https://img.shields.io/badge/Python-3.6+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.8+-ee4c2c.svg)](https://pytorch.org/)

An advanced, production-grade autonomous mobile robot architecture for the NVIDIA JetBot. Moves beyond basic Jupyter notebooks into modular, low-level hardware control, GPU zero-copy camera memory pipelines, and a Finite State Machine (FSM) autonomous parking controller.

---

## Architecture Overview

```
Jetbot-Autonomous-Parking-and-Self-Driving/
├── README.md                           # Master project guide & architecture overview
├── requirements.txt                    # Project dependencies
├── drivers/
│   ├── pca9685_i2c.py                  # Register-level PCA9685 I2C Python driver
│   ├── motor_controller.py             # Differential kinematics motor driver
│   └── c_driver/
│       ├── pca9685.h                   # C Header for I2C register access
│       ├── pca9685.c                   # C implementation for ultra-low latency I2C
│       ├── Makefile                    # Compiles libpca9685.so shared library
│       └── python_wrapper.py           # Py ctypes binding for C driver
├── camera/
│   └── zero_copy_camera.py             # GStreamer NVMM / CUDA zero-copy frame capture
├── ml/
│   ├── dataset_collector.py            # CLI/GUI Multi-class dataset collector
│   ├── dataset.py                      # Custom PyTorch Dataset loader & transforms
│   ├── model.py                        # MobileNetV2 / ResNet18 multi-class classifier
│   └── train.py                        # Standalone training & ONNX export script
├── navigation/
│   └── parking_fsm.py                  # Autonomous Parking Finite State Machine
└── main_autonomous_park.py            # Main entry point combining Driver + Camera + ML + FSM
```

---

## 3 Core Engineering Stages

### Stage A: Low-Level Hardware Drivers (I2C / PCA9685 + TB6612FNG)
- **Direct Register Control**: Communicates directly with the PCA9685 IC (`0x60`) over `/dev/i2c-1` without relying on high-level vendor libraries.
- **Dual Implementation**:
  - Pure Python implementation in [`drivers/pca9685_i2c.py`](file:///C:/Users/samva/Desktop/Jetbot%20Personal%20Project/Jetbot-Autonomous-Parking-and-Self-Driving/drivers/pca9685_i2c.py) using `smbus2`.
  - Native C driver in [`drivers/c_driver/pca9685.c`](file:///C:/Users/samva/Desktop/Jetbot%20Personal%20Project/Jetbot-Autonomous-Parking-and-Self-Driving/drivers/c_driver/pca9685.c) compiled to `libpca9685.so` and invoked via `ctypes`.

#### Compiling the C Shared Library
On the Jetson Nano, navigate to `drivers/c_driver/` and build:
```bash
cd drivers/c_driver
make
```

### Stage B: Zero-Copy CUDA Camera Pipeline
- **Memory Architecture**: On Jetson's SoC architecture, CPU and GPU share physical LPDDR4 memory.
- **Pipeline**: Utilizes GStreamer `nvarguscamerasrc` with `video/x-raw(memory:NVMM)` to map CSI camera frames directly into CUDA memory tensors, bypassing host CPU array allocations.
- Implementation in [`camera/zero_copy_camera.py`](file:///C:/Users/samva/Desktop/Jetbot%20Personal%20Project/Jetbot-Autonomous-Parking-and-Self-Driving/camera/zero_copy_camera.py).

### Stage C: Multi-Class Perception & Autonomous Parking FSM
- **Perception Model**: Transfer learning with MobileNetV2 classifying 5 distinct states:
  1. `path_free`: Open lane ahead.
  2. `obstacle_blocked`: Barrier or obstacle ahead.
  3. `parking_spot_left`: Open spot on left side.
  4. `parking_spot_right`: Open spot on right side.
  5. `parking_spot_occupied`: Occupied parking spot.
- **Finite State Machine**: State transitions handle searching, spot identification, alignment, maneuvering, and safe parking ([`navigation/parking_fsm.py`](file:///C:/Users/samva/Desktop/Jetbot%20Personal%20Project/Jetbot-Autonomous-Parking-and-Self-Driving/navigation/parking_fsm.py)).

---

## Quick Start Guide

### 1. Collect Data for Multi-Class Model
Run the interactive dataset collector:
```bash
python ml/dataset_collector.py
```
Use keys `0` through `4` to capture images for each corresponding class.

### 2. Train the Perception Model
Train MobileNetV2 on your collected dataset and export to ONNX:
```bash
python ml/train.py
```

### 3. Launch Autonomous Parking
Run the full system:
```bash
python main_autonomous_park.py
```

---

## License & Credits
Designed for NVIDIA JetBot hardware on Jetson Nano.
