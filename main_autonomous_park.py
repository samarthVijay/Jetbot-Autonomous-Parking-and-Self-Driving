import os
import sys
import time
import logging
import torch
import torch.nn.functional as F
import numpy as np

os.environ['LD_LIBRARY_PATH'] = '/usr/local/cuda-10.2/lib64:' + os.environ.get('LD_LIBRARY_PATH', '')

from drivers import MotorController
from camera import ZeroCopyCamera
from ml.model import ParkingNet, CLASS_NAMES
from navigation import ParkingFSM, ParkingState, SideDetector

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("AutonomousParkMain")


def preflight_motor_check(motors: MotorController):
    print("\n" + "=" * 55)
    print("  PRE-FLIGHT MOTOR DIRECTION CHECK")
    print("=" * 55)
    input("Press ENTER to test FORWARD movement (1.5 s)...")
    motors.drive_vector(linear_vel=0.20, steering=0.0)
    time.sleep(1.5)
    motors.stop()
    ans = input("Did the JetBot roll straight forward? (y/n): ").strip().lower()
    if ans == 'y':
        print("[PASS] Motor directions verified!\n")
    else:
        print("[WARN] Check motor wiring or direction flags in motor_controller.py!\n")


def main():
    print("\n" + "=" * 65)
    print("  JetBot Autonomous Parking System  (4-class model)")
    print("=" * 65)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Inference device: {device}")
    print(f"Classes: {CLASS_NAMES}\n")

    motors = None
    camera = None

    try:
        logger.info("Initialising motor driver (PCA9685 I2C)...")
        motors = MotorController(i2c_bus=1, i2c_address=0x60)

        ans = input("Run pre-flight motor test? (y/n, default=y): ").strip().lower()
        if ans != 'n':
            preflight_motor_check(motors)

        logger.info("Opening GStreamer CSI camera pipeline...")
        camera = ZeroCopyCamera(width=224, height=224, fps=30)
        if not camera.open():
            logger.error("Failed to open CSI camera!")
            motors.stop()
            sys.exit(1)

        for _ in range(15):
            camera.read_raw()

        logger.info("Loading ParkingNet weights...")
        model = ParkingNet(num_classes=len(CLASS_NAMES), backbone="mobilenet_v2", pretrained=False).to(device)

        model_path = os.path.join(os.path.dirname(__file__), "best_model.pth")
        if os.path.exists(model_path):
            model.load_state_dict(torch.load(model_path, map_location=device))
            logger.info(f"Loaded weights from '{model_path}'")
        else:
            logger.warning(f"'{model_path}' not found — run 'python3 -m ml.train' first.")
            ans = input("Proceed with untrained weights for hardware testing? (y/n): ").strip().lower()
            if ans != 'y':
                camera.release(); motors.stop(); sys.exit(0)

        model.eval()
        d_cuda_tensor = torch.empty((3, 224, 224), dtype=torch.float32, device=device)

        side_detector = SideDetector(
            canny_low=40,
            canny_high=120,
            roi_top_frac=0.40,
            min_ratio=1.30,
        )

        # Set to 'left' or 'right' to always park on one side.
        # None = let ML + CV decide dynamically.
        PARK_SIDE_OVERRIDE = None

        fsm = ParkingFSM(
            motor_controller=motors,
            base_speed=0.20,
            confidence_threshold=0.55,
            voter_window=7,
            park_side_override=PARK_SIDE_OVERRIDE,
            side_detector=side_detector,
        )

        print("\n" + "=" * 65)
        print("  READY")
        print("  1. Place JetBot on your track facing down the main lane.")
        print("  2. Ensure the path ahead is clear.")
        print(f"  3. Park side: {'AUTO (ML + CV)' if PARK_SIDE_OVERRIDE is None else PARK_SIDE_OVERRIDE.upper()}")
        print("=" * 65)
        input("\nPress [ENTER] to start  (Ctrl+C = emergency stop)...")

        print("\nRunning! (Ctrl+C to stop)\n")
        fsm.start_parking_search()

        while True:
            ret, raw_frame = camera.read_raw()
            if not ret or raw_frame is None:
                time.sleep(0.01)
                continue

            import ctypes
            from camera.zero_copy_camera import _CUDA_LIB
            frame_c = np.ascontiguousarray(raw_frame, dtype=np.uint8)
            if not hasattr(main, '_d_input') or main._d_input.shape != frame_c.shape:
                main._d_input = torch.empty(frame_c.shape, dtype=torch.uint8, device=device)
            main._d_input.copy_(torch.from_numpy(frame_c))
            status = _CUDA_LIB.cuda_preprocess(
                ctypes.c_void_p(main._d_input.data_ptr()),
                ctypes.c_void_p(d_cuda_tensor.data_ptr()),
                224, 224
            )
            if status != 0:
                time.sleep(0.01)
                continue

            with torch.no_grad():
                input_batch = d_cuda_tensor.unsqueeze(0)
                outputs     = model(input_batch)
                probs       = F.softmax(outputs, dim=1)
                conf, pred_idx = torch.max(probs, dim=1)

                class_idx  = pred_idx.item()
                confidence = conf.item()
                class_name = CLASS_NAMES[class_idx]

            fsm.update(class_name=class_name, confidence=confidence, raw_frame=raw_frame)

            all_probs = probs[0].cpu().numpy()
            prob_str  = "  ".join(
                f"{CLASS_NAMES[i][:6]}:{all_probs[i]*100:4.1f}%"
                for i in range(len(CLASS_NAMES))
            )
            print(
                f"\r[{fsm.state.name:<14}] "
                f"{class_name:<12} ({confidence*100:4.1f}%)  |  {prob_str}",
                end="", flush=True
            )

            if fsm.state == ParkingState.PARKED:
                print("\n\nJetBot successfully parked!")
                break

            time.sleep(0.02)

    except KeyboardInterrupt:
        logger.info("\nUser stop (Ctrl+C).")
    finally:
        if camera:
            try: camera.release()
            except: pass
        if motors:
            try: motors.stop()
            except: pass
        logger.info("Hardware released. Goodbye.")


if __name__ == "__main__":
    main()
