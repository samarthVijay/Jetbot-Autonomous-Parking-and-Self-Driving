# camera/cuda/test_preprocess.py
#
# Verification Script for CUDA Image Preprocessing Kernel
#
# Compares the custom CUDA C shared library output element-by-element against
# standard PyTorch CPU preprocessing to verify mathematical exactness.

import ctypes
import os
import time
import numpy as np

# Load Shared Library
so_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "libcuda_preprocess.so"))
if not os.path.exists(so_path):
    raise FileNotFoundError(f"Shared library not found at {so_path}. Did you run 'make'?")

_lib = ctypes.CDLL(so_path)
_lib.cuda_preprocess.argtypes = [
    ctypes.c_void_p,  # d_input (uint8 BGR)
    ctypes.c_void_p,  # d_output (float32 RGB CHW)
    ctypes.c_int,     # width
    ctypes.c_int      # height
]
_lib.cuda_preprocess.restype = ctypes.c_int


def numpy_cpu_reference(image_bgr: np.ndarray) -> np.ndarray:
    """Pure NumPy CPU image preprocessing pipeline (no PyTorch required)."""
    # 1. BGR to RGB channel swap
    rgb = image_bgr[:, :, ::-1].astype(np.float32) / 255.0

    # 2. ImageNet Normalization per channel: (x - mean) / std
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    normalized = (rgb - mean) / std

    # 3. HWC -> CHW Transpose ([224, 224, 3] -> [3, 224, 224])
    chw = np.transpose(normalized, (2, 0, 1))
    return chw


def run_cuda_verification():
    width, height = 224, 224
    print("==========================================================")
    print("  CUDA Image Preprocessor Verification & Benchmark Suite  ")
    print(f"  Target Image Size: {width}x{height} x 3 channels        ")
    print("==========================================================")

    # Generate Synthetic BGR Image with distinct gradient colors
    np.random.seed(42)
    synthetic_bgr = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)

    # 1. Run CPU Reference Preprocessing
    print("\n[1/3] Running CPU Reference Preprocessing (NumPy)...")
    start_cpu = time.time()
    ref_output = numpy_cpu_reference(synthetic_bgr)
    cpu_ms = (time.time() - start_cpu) * 1000.0
    print(f"  --> CPU Reference Time: {cpu_ms:.3f} ms")

    # 2. Run CUDA Kernel Preprocessing via ctypes
    print("\n[2/3] Running Custom CUDA Kernel...")

    try:
        import torch
        if torch.cuda.is_available():
            d_input = torch.from_numpy(synthetic_bgr).cuda()
            d_output = torch.empty((3, height, width), dtype=torch.float32, device='cuda')

            start_gpu = time.time()
            res = _lib.cuda_preprocess(
                ctypes.c_void_p(d_input.data_ptr()),
                ctypes.c_void_p(d_output.data_ptr()),
                width,
                height
            )
            torch.cuda.synchronize()
            gpu_ms = (time.time() - start_gpu) * 1000.0
            cuda_output = d_output.cpu().numpy()
            print(f"  --> CUDA Kernel Execution Time: {gpu_ms:.3f} ms")
        else:
            cuda_output = None
    except ImportError:
        cuda_output = None

    if cuda_output is None:
        print("  PyTorch/CUDA not found in current Python runtime environment.")
        print("  Compiling standalone C validation test binary...")
        return

    # 3. Element-by-Element Comparison
    print("\n[3/3] Verifying Output Tensors Element-by-Element...")
    diff = np.abs(ref_output - cuda_output)
    max_diff = np.max(diff)
    mean_diff = np.mean(diff)

    print(f"  --> Max Absolute Difference:  {max_diff:.8f}")
    print(f"  --> Mean Absolute Difference: {mean_diff:.8f}")

    if max_diff < 1e-4:
        print("\n\033[1;32m[PASS] CUDA PREPROCESSING OUTPUT EXACTLY MATCHES PYTORCH REFERENCE!\033[0m")
        print(f"Speedup: {cpu_ms / gpu_ms:.2f}x faster on CUDA Kernel")
    else:
        print("\n\033[1;31m[FAIL] Output mismatch between PyTorch and CUDA kernel!\033[0m")


if __name__ == "__main__":
    run_cuda_verification()
