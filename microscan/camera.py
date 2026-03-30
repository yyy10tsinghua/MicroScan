"""
Camera capture module for digital microscopes.
Wraps OpenCV VideoCapture with resolution configuration.
"""

import os
import cv2
import numpy as np
from typing import Optional, Tuple, List

# Suppress noisy OpenCV backend warnings (DSHOW/MSMF/obsensor)
os.environ.setdefault("OPENCV_LOG_LEVEL", "SILENT")


class MicroscopeCamera:
    """Interface for USB digital microscope cameras."""

    # Backend priority: DSHOW first because MSMF often fails to read inside
    # Qt event loops on Windows. AUTO (which picks MSMF) is last resort.
    _BACKENDS = [
        ("DSHOW", cv2.CAP_DSHOW),
        ("MSMF", cv2.CAP_MSMF),
        ("AUTO", cv2.CAP_ANY),
    ]

    def __init__(self):
        self.cap: Optional[cv2.VideoCapture] = None
        self.device_id: int = 0
        self.backend_name: str = ""

    @property
    def is_open(self) -> bool:
        return self.cap is not None and self.cap.isOpened()

    def open(self, device_id: int = 0, width: int = 0, height: int = 0) -> bool:
        """Open camera device, trying multiple backends.
        If width/height are given, set resolution before first read."""
        self.close()
        self.device_id = device_id

        for name, backend in self._BACKENDS:
            try:
                cap = cv2.VideoCapture(device_id, backend)
            except cv2.error:
                continue
            if not cap.isOpened():
                cap.release()
                continue

            # Set resolution before first read so backend configures properly
            if width > 0 and height > 0:
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

            # Warm-up: try a few reads to let backend settle
            ok = False
            for _ in range(3):
                try:
                    ret, frame = cap.read()
                    if ret and frame is not None and frame.size > 0:
                        ok = True
                        break
                except cv2.error:
                    pass

            if ok:
                self.cap = cap
                self.backend_name = name
                return True
            cap.release()

        return False

    def close(self):
        """Release camera."""
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        self.backend_name = ""

    def read_frame(self) -> Optional[np.ndarray]:
        """Capture a single frame."""
        if not self.is_open:
            return None
        try:
            ret, frame = self.cap.read()
            return frame if ret else None
        except cv2.error:
            return None

    def set_resolution(self, width: int, height: int) -> bool:
        """Set camera resolution by reopening with new settings."""
        if not self.is_open:
            return False
        # Reopen with new resolution to avoid backend instability
        device_id = self.device_id
        return self.open(device_id, width, height)

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
        """List available camera device IDs by trying multiple backends."""
        available = []
        for i in range(max_check):
            for _, backend in MicroscopeCamera._BACKENDS:
                cap = cv2.VideoCapture(i, backend)
                if cap.isOpened():
                    ret, _ = cap.read()
                    cap.release()
                    if ret:
                        available.append(i)
                        break  # found a working backend for this id
                else:
                    cap.release()
        return available
