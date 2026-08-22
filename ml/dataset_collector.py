import os
import sys
import time
import uuid
import cv2

# Ensure repo root is in python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Set LD_LIBRARY_PATH
os.environ['LD_LIBRARY_PATH'] = '/usr/local/cuda-10.2/lib64:' + os.environ.get('LD_LIBRARY_PATH', '')

from camera.zero_copy_camera import ZeroCopyCamera
from ml.model import CLASS_NAMES

def main():
    print("=" * 60)
    print("JetBot Headless Multi-Class Dataset Collector")
    print("=" * 60)
    print("Available Classes:")
    for idx, cname in enumerate(CLASS_NAMES):
        print(f"  [{idx}] {cname}")

    base_dir = os.path.join(os.path.dirname(__file__), "..", "dataset")
    for cname in CLASS_NAMES:
        os.makedirs(os.path.join(base_dir, cname), exist_ok=True)

    camera = ZeroCopyCamera(width=224, height=224, fps=30)
    if not camera.open():
        print("[ERROR] Could not open camera.")
        sys.exit(1)

    print("\nWarmup camera...")
    for _ in range(10):
        camera.read_raw()

    print("\nInteractive Terminal Controls:")
    print("  Type '0'-'4' and press Enter to save current frame to class folder.")
    print("  Type '0 5' to capture 5 images for class 0 with 0.5s delay.")
    print("  Type 'q' and press Enter to quit.")
    print("=" * 60)

    try:
        while True:
            cmd = input("\nEnter class [0-4] (or 'q' to quit): ").strip().lower()
            if cmd == 'q':
                break
            
            parts = cmd.split()
            if not parts:
                continue
                
            class_str = parts[0]
            count = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 1

            if class_str.isdigit() and 0 <= int(class_str) <= 4:
                class_idx = int(class_str)
                class_name = CLASS_NAMES[class_idx]
                target_dir = os.path.join(base_dir, class_name)
                os.makedirs(target_dir, exist_ok=True)

                for i in range(count):
                    ret, frame = camera.read_raw()
                    if ret and frame is not None:
                        filename = f"{uuid.uuid4().hex[:8]}.jpg"
                        filepath = os.path.join(target_dir, filename)
                        cv2.imwrite(filepath, frame)
                        total = len(os.listdir(target_dir))
                        print(f"  [{i+1}/{count}] Saved -> {class_name}/{filename} (Class Total: {total})")
                        if count > 1:
                            time.sleep(0.4)
                    else:
                        print("  [ERROR] Failed to capture frame!")
            else:
                print("  [INVALID] Please enter a number between 0 and 4, or 'q' to quit.")

    finally:
        camera.release()
        print("\nDataset collector closed.")

if __name__ == "__main__":
    main()
