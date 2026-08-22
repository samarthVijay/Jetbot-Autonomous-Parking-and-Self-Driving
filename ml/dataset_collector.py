import os
import sys
import time
import uuid
import cv2

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
os.environ['LD_LIBRARY_PATH'] = '/usr/local/cuda-10.2/lib64:' + os.environ.get('LD_LIBRARY_PATH', '')

from camera.zero_copy_camera import ZeroCopyCamera
from ml.model import CLASS_NAMES

# Collector still saves into the OLD folder names so existing images are preserved.
# dataset.py remaps them to the new 4-class indices at load time.
COLLECT_FOLDERS = {
    0: "path_free",             # -> lane_clear
    1: "parking_spot_left",     # -> spot_left
    2: "parking_spot_right",    # -> spot_right
    3: "obstacle_blocked",      # -> blocked
    4: "parking_spot_occupied", # -> blocked (merged at training time)
}

TIPS = {
    0: "Clear lane ahead, no spots or obstacles visible",
    1: "Gap/opening CLEARLY on LEFT half of frame",
    2: "Gap/opening CLEARLY on RIGHT half of frame",
    3: "Wall / object directly blocking path ahead",
    4: "Spot exists but is filled/occupied",
}


def main():
    print("=" * 65)
    print("  JetBot Dataset Collector  (4-class model, 5 capture folders)")
    print("=" * 65)
    print("\nCapture folders and what to show the camera:\n")
    for k, folder in COLLECT_FOLDERS.items():
        new_class = CLASS_NAMES[min(k, 3)]
        print(f"  [{k}] {folder:<28} -> {new_class}")
        print(f"      Tip: {TIPS[k]}\n")

    base_dir = os.path.join(os.path.dirname(__file__), "..", "dataset")
    for folder in COLLECT_FOLDERS.values():
        os.makedirs(os.path.join(base_dir, folder), exist_ok=True)

    camera = ZeroCopyCamera(width=224, height=224, fps=30)
    if not camera.open():
        print("[ERROR] Could not open camera.")
        sys.exit(1)

    print("Warming up camera...")
    for _ in range(10):
        camera.read_raw()

    print("\nControls:")
    print("  Type '0'-'4' + Enter      -> capture 1 frame")
    print("  Type '0 20' + Enter       -> capture 20 frames with 0.4s delay")
    print("  Type 'count' + Enter      -> show current image counts per class")
    print("  Type 'q' + Enter          -> quit\n")
    print("=" * 65)

    try:
        while True:
            cmd = input("\nCapture class [0-4] (or 'count'/'q'): ").strip().lower()

            if cmd == 'q':
                break

            if cmd == 'count':
                print()
                for k, folder in COLLECT_FOLDERS.items():
                    d = os.path.join(base_dir, folder)
                    n = len([f for f in os.listdir(d) if f.lower().endswith(('.jpg','.jpeg','.png'))]) if os.path.exists(d) else 0
                    new_class = CLASS_NAMES[min(k, 3)]
                    print(f"  [{k}] {folder:<28}: {n:>4} images  (-> {new_class})")
                continue

            parts = cmd.split()
            if not parts or not parts[0].isdigit():
                print("  Invalid input.")
                continue

            class_key = int(parts[0])
            if class_key not in COLLECT_FOLDERS:
                print(f"  Invalid class key. Use 0-4.")
                continue

            count = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 1
            folder = COLLECT_FOLDERS[class_key]
            target_dir = os.path.join(base_dir, folder)

            for i in range(count):
                ret, frame = camera.read_raw()
                if ret and frame is not None:
                    filename = f"{uuid.uuid4().hex[:8]}.jpg"
                    filepath = os.path.join(target_dir, filename)
                    cv2.imwrite(filepath, frame)
                    total = len(os.listdir(target_dir))
                    print(f"  [{i+1}/{count}] {folder}/{filename}  (folder total: {total})")
                    if count > 1:
                        time.sleep(0.4)
                else:
                    print("  [ERROR] Failed to capture frame.")

    finally:
        camera.release()
        print("\nDataset collector closed.")
        print("\nFinal image counts:")
        for k, folder in COLLECT_FOLDERS.items():
            d = os.path.join(base_dir, folder)
            n = len([f for f in os.listdir(d) if f.lower().endswith(('.jpg','.jpeg','.png'))]) if os.path.exists(d) else 0
            print(f"  {folder:<28}: {n} images")


if __name__ == "__main__":
    main()
