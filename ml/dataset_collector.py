import os
import time
import uuid
import cv2
from camera import ZeroCopyCamera
from ml.model import CLASS_NAMES

def main():
    print("=" * 60)
    print("JetBot Multi-Class Dataset Collector")
    print("=" * 60)
    print("Available Classes:")
    for idx, cname in enumerate(CLASS_NAMES):
        print(f"  [{idx}] {cname}")

    base_dir = "dataset"
    for cname in CLASS_NAMES:
        os.makedirs(os.path.join(base_dir, cname), exist_ok=True)

    camera = ZeroCopyCamera()
    print("\nControls:")
    print("  Press '0'-'4' to save current frame to respective class folder.")
    print("  Press 'q' to quit.")

    try:
        while True:
            frame = camera.read_frame_bgr()
            cv2.imshow("Dataset Collector - JetBot", frame)
            
            key = cv2.waitKey(30) & 0xFF
            if key == ord('q'):
                break
            elif ord('0') <= key <= ord('4'):
                class_idx = key - ord('0')
                class_name = CLASS_NAMES[class_idx]
                target_dir = os.path.join(base_dir, class_name)
                
                filename = f"{uuid.uuid4().hex[:8]}.jpg"
                filepath = os.path.join(target_dir, filename)
                cv2.imwrite(filepath, frame)
                count = len(os.listdir(target_dir))
                print(f"Saved snapshot to {class_name} (Total: {count} images)")
    finally:
        camera.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
