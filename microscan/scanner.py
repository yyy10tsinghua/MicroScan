"""
Main scanning controller that orchestrates camera, stitcher, canvas, and overlap analysis.
"""

import numpy as np
from typing import Optional
from .stitcher import ImageStitcher, MatchResult
from .canvas import PanoramaCanvas
from .overlap import OverlapAnalyzer, GuidanceInfo, ScanStatus


class MicroscopeScanner:
    """Orchestrates live microscope scanning with real-time stitching."""

    def __init__(self):
        self.stitcher = ImageStitcher()
        self.canvas = PanoramaCanvas(blend_mode='feather')
        self.analyzer = OverlapAnalyzer()

        self.last_frame: Optional[np.ndarray] = None
        self.current_dx: float = 0.0
        self.current_dy: float = 0.0
        self.last_stitch_dx: float = 0.0
        self.last_stitch_dy: float = 0.0
        self.is_scanning: bool = False
        self.auto_stitch: bool = True
        self._frame_count: int = 0
        # Minimum movement (pixels) since last stitch before allowing next stitch
        self.min_move_since_stitch: float = 30.0

    def start_scanning(self):
        """Start a new scanning session."""
        self.canvas.reset()
        self.last_frame = None
        self.current_dx = 0.0
        self.current_dy = 0.0
        self.last_stitch_dx = 0.0
        self.last_stitch_dy = 0.0
        self.is_scanning = True
        self._frame_count = 0

    def stop_scanning(self):
        """Stop scanning."""
        self.is_scanning = False

    def process_frame(self, frame: np.ndarray) -> GuidanceInfo:
        """Process a new camera frame. Returns guidance for the operator."""
        if not self.is_scanning:
            return GuidanceInfo(status=ScanStatus.IDLE, message="Not scanning")

        self._frame_count += 1

        # First frame: initialize
        if self.last_frame is None:
            self.canvas.initialize(frame)
            self.last_frame = frame.copy()
            return GuidanceInfo(
                status=ScanStatus.INITIALIZED,
                message="Scanning started - move the sample slowly"
            )

        # Find translation from last frame
        match = self.stitcher.find_translation(self.last_frame, frame)

        if match.confidence < 0.1:
            return GuidanceInfo(
                status=ScanStatus.LOST,
                message="Tracking lost - move back slowly",
                confidence=match.confidence
            )

        # Update cumulative position
        self.current_dx += match.dx
        self.current_dy += match.dy

        # Compute overlap with existing panorama
        overlap = self.canvas.compute_frame_overlap(
            frame, self.current_dx, self.current_dy
        )

        # Get guidance
        h, w = frame.shape[:2]
        guidance = self.analyzer.analyze(
            overlap, match.confidence,
            self.current_dx, self.current_dy, w, h
        )

        # Check movement since last stitch
        move_since_stitch = np.sqrt(
            (self.current_dx - self.last_stitch_dx) ** 2 +
            (self.current_dy - self.last_stitch_dy) ** 2
        )

        # Auto-stitch if conditions are met
        if (self.auto_stitch and
                move_since_stitch >= self.min_move_since_stitch and
                self.analyzer.should_stitch(overlap, match.confidence)):
            self.canvas.add_image(frame, self.current_dx, self.current_dy)
            self.last_stitch_dx = self.current_dx
            self.last_stitch_dy = self.current_dy
            guidance.status = ScanStatus.STITCHED
            guidance.message = f"Stitched! ({self.canvas.image_count} images, overlap {overlap:.0%})"

        self.last_frame = frame.copy()
        return guidance

    def force_stitch(self, frame: np.ndarray) -> bool:
        """Force-stitch the current frame regardless of overlap conditions."""
        if not self.is_scanning:
            return False

        if self.last_frame is None:
            self.canvas.initialize(frame)
            self.last_frame = frame.copy()
            return True

        match = self.stitcher.find_translation(self.last_frame, frame)
        self.current_dx += match.dx
        self.current_dy += match.dy

        self.canvas.add_image(frame, self.current_dx, self.current_dy)
        self.last_stitch_dx = self.current_dx
        self.last_stitch_dy = self.current_dy
        self.last_frame = frame.copy()
        return True

    def get_panorama(self) -> Optional[np.ndarray]:
        """Get the current stitched panorama."""
        return self.canvas.get_panorama()

    def get_panorama_size(self):
        """Get panorama dimensions."""
        return self.canvas.get_panorama_size()

    @property
    def image_count(self) -> int:
        return self.canvas.image_count
