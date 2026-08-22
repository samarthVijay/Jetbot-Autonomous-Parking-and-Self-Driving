import os
import sys
import time
import logging
import torch
import torch.nn.functional as F

# Ensure LD_LIBRARY_PATH is set
os.environ['LD_LIBRARY_PATH'] = '/usr/local/cuda-10.2/lib64:' + os.environ.get('LD_LIBRARY_PATH', '')

from drivers import MotorController
from camera import ZeroCopyCamera
from ml.model import ParkingNet, CLASS_NAMES
from navigation import ParkingFSM, ParkingState

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("AutonomousParkMain")


def preflight_motor_check(motors: MotorController):
    print("\n" + "=" * 55)
    print("  PRE-FLIGHT MOTOR DIRECTION CHECK")
    print("=" * 55)
    print("Testing FORWARD movement for 1.5 seconds...")
    print("Check if JetBot rolls STRAIGHT FORWARD.")
    input("Press ENTER to spin motors forward...")
    
    motors.drive_vector(linear_vel=0.20, steering=0.0)
    time.sleep(1.5)
    motors.stop()
    
    print("\nMotors stopped.")
    ans = input("Did the JetBot roll straight forward? (y/n): ").strip().lower()
    if ans == 'y':
        print("[PASS] Motor directions verified!\n")
    else:
        print("[WARN] Motor directions may be inverted. Check motor wiring or direction flags in motor_controller.py!\n")


def main():
    print("\n" + "=" * 65)
    print("  🚗 JetBot Autonomous Parking & Self-Driving System 🚗")
    print("=" * 65)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device Target: {device}")

    motors = None
    camera = None

    try:
        # 1. Initialize Motors
        logger.info("Initializing Low-Level Motor Driver (PCA9685 I2C)...")
        motors = MotorController(i2c_bus=1, i2c_address=0x60)

        # Option for motor check
        ans = input("\nWould you like to run a Pre-Flight Motor Test first? (y/n, default=y): ").strip().lower()
        if ans != 'n':
            preflight_motor_check(motors)

        # 2. Initialize Zero-Copy Camera
        logger.info("Opening GStreamer CSI Camera Pipeline...")
        camera = ZeroCopyCamera(width=224, height=224, fps=30)
        if not camera.open():
            logger.error("Failed to open CSI Camera hardware pipeline!")
            motors.stop()
            sys.exit(1)

        # Warmup camera
        for _ in range(10):
            camera.read_raw()

        # 3. Load Model
        logger.info("Loading ParkingNet Model Weights...")
        model = ParkingNet(num_classes=len(CLASS_NAMES), backbone="mobilenet_v2", pretrained=False).to(device)

        model_path = os.path.join(os.path.dirname(__file__), "best_model.pth")
        if os.path.exists(model_path):
            model.load_state_dict(torch.load(model_path, map_location=device))
            logger.info(f"Successfully loaded trained weights from '{model_path}'!")
        else:
            logger.warning(f"Weight file '{model_path}' not found! Run 'python3 -m ml.train' first.")
            ans = input("Proceed with untrained initial weights for testing? (y/n): ").strip().lower()
            if ans != 'y':
                camera.release()
                motors.stop()
                sys.exit(0)

        model.eval()

        # Allocate persistent GPU output tensor for CUDA kernel preprocessing
        d_cuda_tensor = torch.empty((3, 224, 224), dtype=torch.float32, device=device)

        # 4. Initialize FSM
        fsm = ParkingFSM(motor_controller=motors, base_speed=0.20, confidence_threshold=0.60)

        print("\n" + "=" * 65)
        print("  READY TO LAUNCH AUTONOMOUS PARKING!")
        print("  1. Place JetBot on your track facing down the main lane.")
        print("  2. Make sure the path is clear.")
        print("=" * 65)
        input("\n👉 Press [ENTER] to start Autonomous Parking (or Ctrl+C to exit)... ")

        print("\n🚀 Starting Autonomous Control Loop! (Press Ctrl+C to emergency stop)\n")
        fsm.start_parking_search()

        while True:
            # 1. Run GStreamer capture + CUDA kernel preprocessing directly into GPU tensor
            ok = camera.read_preprocessed_cuda(d_cuda_tensor.data_ptr())
            if not ok:
                time.sleep(0.01)
                continue

            # 2. Add batch dimension: [3, 224, 224] -> [1, 3, 224, 224]
            input_batch = d_cuda_tensor.unsqueeze(0)

            # 3. Forward inference through ParkingNet on GPU
            with torch.no_grad():
                outputs = model(input_batch)
                probs = F.softmax(outputs, dim=1)
                conf, pred_idx = torch.max(probs, dim=1)

                class_idx = pred_idx.item()
                confidence = conf.item()
                class_name = CLASS_NAMES[class_idx]

            # 4. Update FSM Controller
            fsm.update(class_idx=class_idx, class_name=class_name, confidence=confidence)

            # 5. Live status printing
            print(f"\r[FSM: {fsm.state.name:<16}] Perception: {class_name:<22} ({confidence*100:5.1f}%)", end="", flush=True)

            # Stop if parked
            if fsm.state == ParkingState.PARKED:
                print("\n\n🎉 JETBOT SUCCESSFULLY PARKED! Mission accomplished.")
                break

            time.sleep(0.02)  # ~50Hz loop rate

    except KeyboardInterrupt:
        logger.info("\n\nUser requested exit (Ctrl+C). Stopping hardware...")
    finally:
        if camera:
            try:
                camera.release()
            except:
                pass
        if motors:
            try:
                motors.stop()
            except:
                pass
        logger.info("Hardware released safely. Goodbye!")


if __name__ == "__main__":
    main()


if __name__ == "__main__":
    main()
