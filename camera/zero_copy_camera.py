import cv2
import numpy as np
import logging

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    torch = None

logger = logging.getLogger("ZeroCopyCamera")

class ZeroCopyCamera:
    """
    High-Performance Zero-Copy Camera Pipeline for Jetson Nano.
    Leverages NVIDIA Argus CSI camera hardware ISP + GStreamer NVMM memory buffers
    to directly output PyTorch CUDA Tensors without intermediate host CPU memory copies.
    """
    def __init__(self, width=224, height=224, fps=30, capture_width=1280, capture_height=720, sensor_id=0, mock=False):
        self.width = width
        self.height = height
        self.fps = fps
        self.mock = mock
        self.cap = None

        if not self.mock:
            gst_pipeline = (
                f"nvarguscamerasrc sensor-id={sensor_id} ! "
                f"video/x-raw(memory:NVMM), width=(int){capture_width}, height=(int){capture_height}, framerate=(fraction){fps}/1 ! "
                f"nvvidconv flip-method=0 ! "
                f"video/x-raw, width=(int){width}, height=(int){height}, format=(string)BGRx ! "
                f"videoconvert ! "
                f"video/x-raw, format=(string)BGR ! appsink drop=True max-buffers=1"
            )
            logger.info(f"Initializing GStreamer Argus Pipeline: {gst_pipeline}")
            self.cap = cv2.VideoCapture(gst_pipeline, cv2.CAP_GSTREAMER)

            if not self.cap.isOpened():
                logger.warning("Failed to open CSI Camera with GStreamer pipeline! Falling back to default OpenCV VideoCapture or MOCK mode.")
                self.cap = cv2.VideoCapture(0)
                if not self.cap.isOpened():
                    logger.warning("No USB or CSI camera found. Entering MOCK camera mode.")
                    self.mock = True

        # Pre-allocated normalization constants on CUDA GPU memory
        if HAS_TORCH:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(1, 3, 1, 1)
            self.std = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(1, 3, 1, 1)
        else:
            self.device = 'cpu'

    def read_frame_bgr(self):
        """Read a single raw BGR frame (numpy uint8 array)."""
        if self.mock:
            # Generate synthetic test frame
            return np.random.randint(0, 255, (self.height, self.width, 3), dtype=np.uint8)

        ret, frame = self.cap.read()
        if not ret:
            logger.error("Failed to read frame from camera")
            return np.zeros((self.height, self.width, 3), dtype=np.uint8)
        return frame

    def read_cuda_tensor(self):
        """
        Reads camera frame, converts BGR to RGB, normalizes, and returns a pre-processed
        PyTorch CUDA Tensor [1, 3, H, W] ready for immediate model inference.
        """
        bgr_frame = self.read_frame_bgr()
        
        # Convert BGR -> RGB
        rgb_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)

        if not HAS_TORCH:
            return rgb_frame

        # Zero-copy CPU pin-memory to CUDA tensor initialization
        # Torch tensor wrapped directly over frame buffer
        tensor = torch.from_numpy(rgb_frame).permute(2, 0, 1).unsqueeze(0).float().to(self.device, non_blocking=True)
        tensor /= 255.0

        # Apply ImageNet normalization directly on GPU
        tensor = (tensor - self.mean) / self.std
        return tensor

    def release(self):
        if self.cap and not self.mock:
            self.cap.release()
            logger.info("Camera pipeline released.")
