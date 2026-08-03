import time
import logging
import torch
import torch.nn.functional as F
from drivers import MotorController
from camera import ZeroCopyCamera
from ml.model import ParkingNet, CLASS_NAMES
from navigation import ParkingFSM, ParkingState

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("AutonomousParkMain")


def main():
    print("=" * 65)
    print(" JetBot Autonomous Parking & Low-Level Control System ")
    print("=" * 65)

    # 1. Initialize Low-Level Motor Driver (PCA9685 I2C)
    logger.info("Initializing Low-Level Motor Driver...")
    motors = MotorController(i2c_bus=1, i2c_address=0x60)

    # 2. Initialize Zero-Copy CUDA Camera Pipeline
    logger.info("Initializing Zero-Copy Camera Pipeline...")
    camera = ZeroCopyCamera(width=224, height=224, fps=30)

    # 3. Load Trained Multi-Class Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ParkingNet(num_classes=len(CLASS_NAMES), backbone="mobilenet_v2", pretrained=False).to(device)

    model_path = "best_model.pth"
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        logger.info(f"Loaded trained model weights from '{model_path}'")
    except Exception as e:
        logger.warning(f"Could not load '{model_path}' ({e}). Running with untrained initial weights for testing.")

    model.eval()

    # 4. Initialize Finite State Machine Controller
    fsm = ParkingFSM(motor_controller=motors, base_speed=0.20)
    fsm.start_parking_search()

    logger.info("Starting Autonomous Control Loop... Press Ctrl+C to terminate.")

    try:
        while True:
            # Capture camera frame as PyTorch CUDA Tensor (Zero-Copy pipeline)
            tensor_frame = camera.read_cuda_tensor()

            # Execute forward inference
            with torch.no_grad():
                outputs = model(tensor_frame)
                probs = F.softmax(outputs, dim=1)
                conf, pred_class_idx = torch.max(probs, dim=1)

                class_idx = pred_class_idx.item()
                confidence = conf.item()
                class_name = CLASS_NAMES[class_idx]

            # Update FSM controller state
            fsm.update(class_idx=class_idx, class_name=class_name, confidence=confidence)

            # Log current state
            print(f"\rState: {fsm.state.name:<16} | Perception: {class_name:<22} ({confidence*100:.1f}%)", end="", flush=True)

            time.sleep(0.02)  # ~50Hz loop rate

    except KeyboardInterrupt:
        logger.info("\nTermination requested by user (Ctrl+C). Stopping hardware...")
    finally:
        fsm.stop()
        camera.release()
        motors.stop()
        logger.info("System safely shut down.")


if __name__ == "__main__":
    main()
