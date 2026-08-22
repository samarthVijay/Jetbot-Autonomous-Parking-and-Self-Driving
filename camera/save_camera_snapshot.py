# camera/save_camera_snapshot.py
#
# Captures a live frame from the CSI camera, runs custom CUDA preprocessing,
# and saves both the raw BGR image and the CUDA preprocessed tensor (denormalized back to RGB)
# to disk as JPEG files for visual inspection.

import os
import sys
import cv2
import numpy as np

# Ensure repo root is in python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Ensure CUDA library path is set
os.environ['LD_LIBRARY_PATH'] = '/usr/local/cuda-10.2/lib64:' + os.environ.get('LD_LIBRARY_PATH', '')

try:
    import torch
except ImportError:
    print("[ERROR] PyTorch is required.")
    sys.exit(1)

from camera.zero_copy_camera import ZeroCopyCamera


def denormalize_imagenet(tensor_chw: torch.Tensor) -> np.ndarray:
    """
    Reverses ImageNet normalization ((x - mean) / std) -> uint8 RGB [0, 255]
    Input shape: [3, H, W] tensor on GPU
    Returns: uint8 numpy array [H, W, 3] BGR for cv2.imwrite
    """
    mean = torch.tensor([0.485, 0.456, 0.406], device=tensor_chw.device).view(3, 1, 1)
    std  = torch.tensor([0.229, 0.224, 0.225], device=tensor_chw.device).view(3, 1, 1)

    # Reverse normalization: x = (norm * std) + mean
    rgb_float = (tensor_chw * std) + mean
    rgb_float = torch.clamp(rgb_float, 0.0, 1.0) * 255.0

    # CHW [3, H, W] -> HWC [H, W, 3]
    hwc_rgb = rgb_float.permute(1, 2, 0).cpu().numpy().astype(np.uint8)

    # RGB -> BGR for OpenCV saving
    hwc_bgr = cv2.cvtColor(hwc_rgb, cv2.COLOR_RGB2BGR)
    return hwc_bgr


def main():
    print("=" * 60)
    print("   CSI Camera Live Frame + CUDA Preprocess Snapshot Tool   ")
    print("=" * 60)

    out_dir = os.path.join(os.path.dirname(__file__), "snapshots")
    os.makedirs(out_dir, exist_ok=True)

    cam = ZeroCopyCamera(width=224, height=224, fps=30)
    if not cam.open():
        print("[ERROR] Failed to open CSI camera.")
        sys.exit(1)

    print("\n[1/3] Warming up camera sensor...")
    for _ in range(15):
        cam.read_raw()

    # 1. Capture Raw BGR Frame
    print("[2/3] Capturing live raw frame...")
    ret, raw_frame = cam.read_raw()
    if not ret or raw_frame is None:
        print("[ERROR] Failed to capture frame.")
        cam.release()
        sys.exit(1)

    raw_path = os.path.join(out_dir, "raw_camera_frame.jpg")
    cv2.imwrite(raw_path, raw_frame)
    print(f"  --> Saved Raw Frame: {raw_path}")

    # 2. Run CUDA Preprocessing Kernel into GPU Tensor
    print("\n[3/3] Running Custom CUDA Preprocessing Kernel...")
    d_output = torch.empty((3, 224, 224), dtype=torch.float32, device='cuda')
    ok = cam.read_preprocessed_cuda(d_output.data_ptr())
    torch.cuda.synchronize()

    if not ok:
        print("[ERROR] CUDA preprocessing failed.")
        cam.release()
        sys.exit(1)

    # 3. Denormalize CUDA tensor back to image to visually verify pixel values
    denorm_bgr = denormalize_imagenet(d_output)
    cuda_path = os.path.join(out_dir, "cuda_preprocessed_frame.jpg")
    cv2.imwrite(cuda_path, denorm_bgr)
    print(f"  --> Saved CUDA Preprocessed Frame: {cuda_path}")

    cam.release()

    print("\n" + "=" * 60)
    print("   SUCCESS! Both snapshots saved to disk.")
    print("   Tensor Stats from CUDA Kernel:")
    print(f"     Shape:  {d_output.shape}")
    print(f"     Device: {d_output.device}")
    print(f"     Range:  [{d_output.min().item():.4f}, {d_output.max().item():.4f}]")
    print(f"   Files to SCP to your PC:")
    print(f"     1. {raw_path}")
    print(f"     2. {cuda_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
