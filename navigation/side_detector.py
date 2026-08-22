import logging
from typing import Optional
import numpy as np

logger = logging.getLogger("SideDetector")


class SideDetector:
    """
    Deterministic left/right spot side detector using edge density analysis.

    A parking spot shows as lower edge density on one side of the frame
    (the gap/opening has less "stuff" in it than the occupied/wall side).

    Usage:
        detector = SideDetector()
        side = detector.detect(raw_bgr_frame)
        # Returns 'spot_left', 'spot_right', or None (ambiguous)
    """

    def __init__(
        self,
        canny_low: int = 40,
        canny_high: int = 120,
        roi_top_frac: float = 0.40,  # ignore top 40% of frame
        min_ratio: float = 1.30,      # one side must have 30% more edges to declare
    ):
        self.canny_low    = canny_low
        self.canny_high   = canny_high
        self.roi_top_frac = roi_top_frac
        self.min_ratio    = min_ratio
        self._cv2 = None

    def _get_cv2(self):
        if self._cv2 is None:
            import cv2
            self._cv2 = cv2
        return self._cv2

    def detect(self, bgr_frame) -> Optional[str]:
        if bgr_frame is None:
            return None

        try:
            cv2 = self._get_cv2()
        except ImportError:
            logger.warning("OpenCV not available — SideDetector disabled.")
            return None

        h, w = bgr_frame.shape[:2]
        roi_top = int(h * self.roi_top_frac)

        gray  = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2GRAY)
        blur  = cv2.GaussianBlur(gray, (5, 5), 0)
        roi   = blur[roi_top:, :]
        edges = cv2.Canny(roi, self.canny_low, self.canny_high)

        mid         = w // 2
        left_edges  = int(np.sum(edges[:, :mid]))
        right_edges = int(np.sum(edges[:, mid:]))

        if left_edges == 0 and right_edges == 0:
            return None
        if left_edges == 0:
            return "spot_left"
        if right_edges == 0:
            return "spot_right"

        ratio = right_edges / left_edges

        if ratio >= self.min_ratio:
            return "spot_left"   # right has more edges → gap is left
        elif (1.0 / ratio) >= self.min_ratio:
            return "spot_right"  # left has more edges → gap is right
        else:
            return None  # ambiguous — trust ML

    def debug_overlay(self, bgr_frame):
        """Returns annotated frame for visual debugging during development."""
        try:
            cv2 = self._get_cv2()
        except ImportError:
            return bgr_frame

        h, w = bgr_frame.shape[:2]
        roi_top = int(h * self.roi_top_frac)
        mid = w // 2

        gray  = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2GRAY)
        blur  = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blur[roi_top:, :], self.canny_low, self.canny_high)

        out      = bgr_frame.copy()
        edge_vis = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        out[roi_top:] = cv2.addWeighted(out[roi_top:], 0.5, edge_vis, 0.5, 0)
        cv2.line(out, (mid, roi_top), (mid, h), (0, 255, 0), 1)
        cv2.line(out, (0, roi_top), (w, roi_top), (0, 200, 255), 1)

        result = self.detect(bgr_frame)
        cv2.putText(out, result or "ambiguous", (5, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        return out
