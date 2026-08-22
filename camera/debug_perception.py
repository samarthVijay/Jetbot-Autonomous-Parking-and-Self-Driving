# camera/debug_perception.py
#
# Diagnostic tool: Runs zero-copy CUDA camera pipeline + ParkingNet inference
# with MOTORS DISABLED. Prints top prediction and full 5-class softmax probabilities
# in real time so you can inspect model perception while moving JetBot by hand.

import os
import sys
import time
import torch
import torch.nn.functional as F

# Ensure LD_LIBRARY_PATH is set
os.environ['LD_LIBRARY_PATH'] = '/usr/local/cuda-10.2/lib64:' + os.environ.get('LD_LIBRARY_PATH', '')

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from camera.zero_copy_camera import ZeroCopyCamera
from ml.model import ParkingNet, CLASS_NAMES


def main():
    print("\n" + "=" * 65)
    print("  🔍 REAL-TIME PERCEPTION DEBUGGER (MOTORS DISABLED) 🔍")
    print("=" * 65)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Initialize Camera
    camera = ZeroCopyCamera(width=224, height=224, fps=30)
    if not camera.open():
        print("[ERROR] Could not open CSI Camera.")
        sys.exit(1)

    for _ in range(10):
        camera.read_raw()

    # Load Model
    model = ParkingNet(num_classes=len(CLASS_NAMES), backbone="mobilenet_v2", pretrained=False).to(device)
    model_path = os.path.join(os.path.dirname(__file__), "..", "best_model.pth")

    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"[LOADED] Weights from '{model_path}'\n")
    else:
        print(f"[ERROR] Weight file '{model_path}' not found! Train model first.")
        camera.release()
        sys.exit(1)

    model.eval()
    d_cuda_tensor = torch.empty((3, 224, 224), dtype=torch.float32, device=device)

    print("Move JetBot around your track by hand and watch live predictions below.")
    print("Press Ctrl+C to quit.\n")
    print("=" * 65)

    try:
        while True:
            ok = camera.read_preprocessed_cuda(d_cuda_tensor.data_ptr())
            if not ok:
                time.sleep(0.01)
                continue

            input_batch = d_cuda_tensor.unsqueeze(0)

            with torch.no_grad():
                outputs = model(input_batch)
                probs = F.softmax(outputs, dim=1)[0]
                conf, pred_idx = torch.max(probs, dim=0)

            idx = pred_idx.item()
            pred_name = CLASS_NAMES[idx]
            top_conf = conf.item() * 100.0

            # Format string for all 5 class percentages
            p_free  = probs[0].item() * 100.0
            p_block = probs[1].item() * 100.0
            p_left  = probs[2].item() * 100.0
            p_right = probs[3].item() * 100.0
            p_occ   = probs[4].item() * 100.0

            print(f"\rPRED: {pred_name:<20} ({top_conf:5.1f}%) | Free:{p_free:4.1f}% | Block:{p_block:4.1f}% | Left:{p_left:4.1f}% | Right:{p_right:4.1f}% | Occ:{p_occ:4.1f}%", end="", flush=True)

            time.sleep(0.05)

    except KeyboardInterrupt:
        print("\n\nDebugger stopped.")
    finally:
        camera.release()


if __name__ == "__main__":
    main()
