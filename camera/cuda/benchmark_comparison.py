# camera/cuda/benchmark_comparison.py
#
# Benchmark Comparison Suite:
# 1. CPU Preprocessing (NumPy BGR->RGB, /255, ImageNet norm, HWC->CHW)
# 2. PyTorch Standard GPU Preprocessing (CPU Host memory -> GPU copy -> PyTorch transforms)
# 3. Custom Fused CUDA Kernel (Direct GPU execution on persistent unified buffer)

import os
import sys
import time
import ctypes
import numpy as np

# Ensure CUDA library path is set
os.environ['LD_LIBRARY_PATH'] = '/usr/local/cuda-10.2/lib64:' + os.environ.get('LD_LIBRARY_PATH', '')

try:
    import torch
except ImportError:
    print("[ERROR] PyTorch is required to run this benchmark.")
    sys.exit(1)

# Load CUDA Preprocessing Library
_SO_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "libcuda_preprocess.so"))
if not os.path.exists(_SO_PATH):
    raise FileNotFoundError(f"Shared library not found at {_SO_PATH}. Run 'make' in camera/cuda first.")

_lib = ctypes.CDLL(_SO_PATH)
_lib.cuda_preprocess.argtypes = [
    ctypes.c_void_p,  # d_input (uint8 BGR)
    ctypes.c_void_p,  # d_output (float32 RGB CHW)
    ctypes.c_int,     # width
    ctypes.c_int      # height
]
_lib.cuda_preprocess.restype = ctypes.c_int


def benchmark_cpu(image_bgr: np.ndarray, iterations: int = 100):
    """Approach 1: Pure CPU NumPy Preprocessing"""
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    times = []
    for _ in range(iterations):
        t0 = time.time()
        # BGR -> RGB & convert float
        rgb = image_bgr[:, :, ::-1].astype(np.float32) / 255.0
        # Normalize
        norm = (rgb - mean) / std
        # HWC -> CHW
        chw = np.transpose(norm, (2, 0, 1))
        times.append((time.time() - t0) * 1000.0)

    avg_ms = np.mean(times)
    fps = 1000.0 / avg_ms
    return avg_ms, fps, chw


def benchmark_pytorch_gpu(image_bgr: np.ndarray, iterations: int = 100):
    """Approach 2: Standard PyTorch GPU Preprocessing (CPU -> GPU copy + PyTorch operations)"""
    mean_t = torch.tensor([0.485, 0.456, 0.406], device='cuda', dtype=torch.float32).view(3, 1, 1)
    std_t  = torch.tensor([0.229, 0.224, 0.225], device='cuda', dtype=torch.float32).view(3, 1, 1)

    times = []
    for _ in range(iterations):
        t0 = time.time()
        # 1. H2D Copy (CPU Host -> GPU Device)
        t_gpu = torch.from_numpy(image_bgr).cuda()
        # 2. BGR -> RGB & float conversion
        t_rgb = t_gpu[:, :, [2, 1, 0]].permute(2, 0, 1).float() / 255.0
        # 3. Normalize
        t_norm = (t_rgb - mean_t) / std_t
        torch.cuda.synchronize()
        times.append((time.time() - t0) * 1000.0)

    avg_ms = np.mean(times)
    fps = 1000.0 / avg_ms
    return avg_ms, fps, t_norm.cpu().numpy()


def benchmark_custom_cuda_kernel(image_bgr: np.ndarray, iterations: int = 100):
    """Approach 3: Custom Fused CUDA Kernel (Fused single-pass CUDA execution)"""
    height, width, _ = image_bgr.shape
    d_input = torch.empty((height, width, 3), dtype=torch.uint8, device='cuda')
    d_output = torch.empty((3, height, width), dtype=torch.float32, device='cuda')

    times = []
    for _ in range(iterations):
        t0 = time.time()
        # Transfer frame to persistent GPU buffer
        d_input.copy_(torch.from_numpy(image_bgr))
        # Launch Fused CUDA Kernel
        _lib.cuda_preprocess(
            ctypes.c_void_p(d_input.data_ptr()),
            ctypes.c_void_p(d_output.data_ptr()),
            width, height
        )
        torch.cuda.synchronize()
        times.append((time.time() - t0) * 1000.0)

    avg_ms = np.mean(times)
    fps = 1000.0 / avg_ms
    return avg_ms, fps, d_output.cpu().numpy()


def run_benchmarks():
    width, height = 224, 224
    iterations = 100
    np.random.seed(42)
    synthetic_bgr = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)

    print("\n" + "=" * 65)
    print("      JETSON NANO CAMERA PREPROCESSING BENCHMARK SHOWDOWN     ")
    print(f"      Image Size: {width}x{height}x3  |  Iterations: {iterations}")
    print("=" * 65)

    print("\n[1/3] Benchmarking CPU NumPy Preprocessing...")
    cpu_ms, cpu_fps, cpu_out = benchmark_cpu(synthetic_bgr, iterations)

    print("[2/3] Benchmarking Standard PyTorch GPU Pipeline...")
    torch_ms, torch_fps, torch_out = benchmark_pytorch_gpu(synthetic_bgr, iterations)

    print("[3/3] Benchmarking Custom Fused CUDA Kernel Pipeline...")
    cuda_ms, cuda_fps, cuda_out = benchmark_custom_cuda_kernel(synthetic_bgr, iterations)

    # Verification
    diff_torch = np.max(np.abs(cpu_out - torch_out))
    diff_cuda = np.max(np.abs(cpu_out - cuda_out))

    print("\n" + "=" * 65)
    print("                     BENCHMARK RESULTS SUMMARY                ")
    print("=" * 65)
    print(f" Approach 1: CPU NumPy Preprocessing")
    print(f"   --> Latency:    {cpu_ms:6.2f} ms / frame")
    print(f"   --> Throughput: {cpu_fps:6.1f} FPS")
    print()
    print(f" Approach 2: Standard PyTorch GPU Ops (cuda + permute + div + sub)")
    print(f"   --> Latency:    {torch_ms:6.2f} ms / frame")
    print(f"   --> Throughput: {torch_fps:6.1f} FPS")
    print(f"   --> Speedup:    {cpu_ms / torch_ms:6.2f}x vs CPU")
    print()
    print(f" Approach 3: Custom Fused CUDA Kernel (Single-pass GPU Execution)")
    print(f"   --> Latency:    {cuda_ms:6.2f} ms / frame")
    print(f"   --> Throughput: {cuda_fps:6.1f} FPS")
    print(f"   --> Speedup:    {cpu_ms / cuda_ms:6.2f}x vs CPU  ({torch_ms / cuda_ms:.2f}x vs PyTorch GPU)")
    print("=" * 65)
    print(f" Numerical Accuracy (Max Abs Diff vs CPU Reference):")
    print(f"   --> Standard PyTorch GPU: {diff_torch:.8f}")
    print(f"   --> Custom CUDA Kernel:   {diff_cuda:.8f}")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    run_benchmarks()
