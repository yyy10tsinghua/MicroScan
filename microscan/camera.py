"""
Camera capture module for digital microscopes.
Wraps OpenCV VideoCapture with resolution configuration.
"""

import cv2
import numpy as np
from typing import Optional, Tuple, List


class MicroscopeCamera:
    """Interface for USB digital microscope cameras."""

    def __init__(self):
        self.cap: Optional[cv2.VideoCapture] = None
        self.device_id: int = 0

    @property
    def is_open(self) -> bool:
        return self.cap is not None and self.cap.isOpened()

    def open(self, device_id: int = 0) -> bool:
        """Open camera device."""
        self.close()
        self.device_id = device_id
        self.cap = cv2.VideoCapture(device_id, cv2.CAP_DSHOW)
        if not self.cap.isOpened():
            self.cap = cv2.VideoCapture(device_id)
        return self.is_open

    def close(self):
        """Release camera."""
        if self.cap is not None:
            self.cap.release()
            self.cap = None

    def read_frame(self) -> Optional[np.ndarray]:
        """Capture a single frame."""
        if not self.is_open:
            return None
        ret, frame = self.cap.read()
        return frame if ret else None

    def set_resolution(self, width: int, height: int) -> bool:
        """Set camera resolution."""
        if not self.is_open:
            return False
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        return True

    def get_resolution(self) -> Tuple[int, int]:
        """Get current resolution (width, height)."""
        if not self.is_open:
            return (0, 0)
        w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        return (w, h)

    def set_property(self, prop_id: int, value: float):
        """Set a camera property."""
        if self.is_open:
            self.cap.set(prop_id, value)

    @staticmethod
    def list_cameras(max_check: int = 5) -> List[int]:
        """List available camera device IDs."""
        available = []
        for i in range(max_check):
            cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
            if cap.isOpened():
                available.append(i)
                cap.release()
            else:
                cap = cv2.VideoCapture(i)
                if cap.isOpened():
                    available.append(i)
                    cap.release()
        return available
