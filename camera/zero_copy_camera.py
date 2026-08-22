# camera/zero_copy_camera.py
#
# Hardware-Accelerated Zero-Copy CSI Camera Pipeline for NVIDIA Jetson Nano
# Integrates GStreamer nvarguscamerasrc + Custom CUDA Preprocessing Kernel (libcuda_preprocess.so)

import os
import ctypes
import cv2
import numpy as np

try:
    import torch
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False

# Load compiled CUDA preprocessing shared library
_SO_PATH = os.path.join(os.path.dirname(__file__), "cuda", "libcuda_preprocess.so")
_CUDA_LIB = None

if os.path.exists(_SO_PATH):
    try:
        _CUDA_LIB = ctypes.CDLL(_SO_PATH)
        _CUDA_LIB.cuda_preprocess.argtypes = [
            ctypes.c_void_p,  # d_input (uint8 BGR)
            ctypes.c_void_p,  # d_output (float32 RGB CHW)
            ctypes.c_int,     # width
            ctypes.c_int      # height
        ]
        _CUDA_LIB.cuda_preprocess.restype = ctypes.c_int
    except Exception as e:
        print(f"[WARN] Failed to load libcuda_preprocess.so: {e}")


def gstreamer_pipeline(
    capture_width=1280,
    capture_height=720,
    display_width=224,
    display_height=224,
    framerate=30,
    flip_method=0
):
    """
    Builds Jetson Nano NVMM Hardware-Accelerated GStreamer Pipeline String.
    Uses nvarguscamerasrc (ISP) -> nvvidconv (HW Scaler) -> video/x-raw BGR
    """
    return (
        f"nvarguscamerasrc ! "
        f"video/x-raw(memory:NVMM), width=(int){capture_width}, height=(int){capture_height}, "
        f"format=(string)NV12, framerate=(fraction){framerate}/1 ! "
        f"nvvidconv flip-method={flip_method} ! "
        f"video/x-raw, width=(int){display_width}, height=(int){display_height}, format=(string)BGRx ! "
        f"videoconvert ! "
        f"video/x-raw, format=(string)BGR ! "
        f"appsink drop=true sync=false"
    )


class ZeroCopyCamera:
    """
    Zero-Copy CSI Camera Interface for Jetbot Autonomous Parking.
    
    Captures frames using GStreamer hardware acceleration, passes raw frame pointers
    to the custom CUDA preprocessing kernel, and returns preprocessed PyTorch GPU tensors.
    """
    def __init__(self, width=224, height=224, fps=30, flip_method=0):
        self.width = width
        self.height = height
        self.fps = fps
        self.flip_method = flip_method
        self.pipeline = gstreamer_pipeline(
            display_width=self.width,
            display_height=self.height,
            framerate=self.fps,
            flip_method=self.flip_method
        )
        self.cap = None
        self._is_opened = False

    def open(self):
        """Initializes and opens the GStreamer hardware camera pipeline."""
        print(f"[INFO] Opening GStreamer CSI Camera Pipeline ({self.width}x{self.height} @ {self.fps}fps)...")
        self.cap = cv2.VideoCapture(self.pipeline, cv2.CAP_GSTREAMER)
        
        if not self.cap.isOpened():
            print("[WARN] GStreamer pipeline failed to open. Falling back to V4L2 default camera index 0...")
            self.cap = cv2.VideoCapture(0)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)

        self._is_opened = self.cap.isOpened()
        if self._is_opened:
            print("[SUCCESS] Camera pipeline online!")
        else:
            print("[ERROR] Failed to initialize camera hardware.")
        return self._is_opened

    def read_raw(self):
        """Captures a raw BGR uint8 frame [224, 224, 3] from the hardware pipeline."""
        if not self._is_opened:
            raise RuntimeError("Camera is not opened. Call open() first.")
        ret, frame = self.cap.read()
        if not ret or frame is None:
            return False, None
        return True, frame

    def read_preprocessed_cuda(self, d_output_ptr):
        """
        Captures a frame and runs it through our custom CUDA preprocessing kernel.
        
        The frame is transferred to GPU memory via PyTorch before the kernel
        reads it, since CUDA kernels cannot access CPU memory pointers.
        
        Args:
            d_output_ptr: Memory address (int/void_p) of pre-allocated GPU float32 tensor [3, 224, 224]
        
        Returns:
            bool: Success flag
        """
        ret, frame = self.read_raw()
        if not ret:
            return False

        if _CUDA_LIB is None:
            raise RuntimeError("CUDA preprocessing library libcuda_preprocess.so not loaded!")
        if not _HAS_TORCH:
            raise RuntimeError("PyTorch is required for GPU memory transfer!")

        # Transfer frame to GPU (reuse buffer to avoid repeated allocation)
        frame_contiguous = np.ascontiguousarray(frame, dtype=np.uint8)
        if not hasattr(self, '_d_input') or self._d_input.shape != frame_contiguous.shape:
            self._d_input = torch.empty(frame_contiguous.shape, dtype=torch.uint8, device='cuda')
        self._d_input.copy_(torch.from_numpy(frame_contiguous))
        input_ptr = self._d_input.data_ptr()

        # Execute CUDA Kernel (both pointers now point to GPU memory)
        status = _CUDA_LIB.cuda_preprocess(
            ctypes.c_void_p(input_ptr),
            ctypes.c_void_p(d_output_ptr),
            self.width,
            self.height
        )

        return status == 0

    def release(self):
        """Releases camera hardware resources."""
        if self.cap:
            self.cap.release()
            self._is_opened = False
            print("[INFO] Camera hardware released.")

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()
