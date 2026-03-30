"""
Overlap analysis and movement guidance for live scanning.
Computes overlap metrics and generates directional guidance.
"""

import numpy as np
from dataclasses import dataclass
from enum import Enum
from typing import Optional


class ScanStatus(Enum):
    IDLE = "idle"
    INITIALIZED = "initialized"
    TRACKING = "tracking"
    STITCHED = "stitched"
    GOOD_OVERLAP = "good_overlap"
    TOO_MUCH_OVERLAP = "too_much_overlap"
    TOO_LITTLE_OVERLAP = "too_little_overlap"
    LOST = "lost"


class Direction(Enum):
    NONE = ""
    LEFT = "← Move Left"
    RIGHT = "→ Move Right"
    UP = "↑ Move Up"
    DOWN = "↓ Move Down"
    LEFT_UP = "↖ Move Left-Up"
    LEFT_DOWN = "↙ Move Left-Down"
    RIGHT_UP = "↗ Move Right-Up"
    RIGHT_DOWN = "↘ Move Right-Down"


@dataclass
class GuidanceInfo:
    """Guidance information for the operator."""
    status: ScanStatus = ScanStatus.IDLE
    overlap: float = 0.0
    direction: Direction = Direction.NONE
    message: str = ""
    confidence: float = 0.0
    dx: float = 0.0
    dy: float = 0.0

    @property
    def color(self):
        """Color for GUI overlay: (B, G, R)."""
        if self.status == ScanStatus.GOOD_OVERLAP:
            return (0, 255, 0)       # Green
        elif self.status == ScanStatus.STITCHED:
            return (0, 255, 128)     # Light green
        elif self.status in (ScanStatus.TOO_MUCH_OVERLAP, ScanStatus.TOO_LITTLE_OVERLAP):
            return (0, 200, 255)     # Yellow
        elif self.status == ScanStatus.LOST:
            return (0, 0, 255)       # Red
        elif self.status == ScanStatus.TRACKING:
            return (255, 200, 0)     # Cyan
        return (200, 200, 200)       # Gray


class OverlapAnalyzer:
    """Analyzes overlap and generates scanning guidance."""

    def __init__(self,
                 min_overlap: float = 0.15,
                 max_overlap: float = 0.70,
                 ideal_overlap_low: float = 0.20,
                 ideal_overlap_high: float = 0.50,
                 stitch_threshold: float = 0.15):
        self.min_overlap = min_overlap
        self.max_overlap = max_overlap
        self.ideal_overlap_low = ideal_overlap_low
        self.ideal_overlap_high = ideal_overlap_high
        # Minimum new-content ratio to trigger stitch
        self.stitch_threshold = stitch_threshold

    def analyze(self, overlap: float, confidence: float,
                dx: float, dy: float,
                frame_w: int, frame_h: int) -> GuidanceInfo:
        """Analyze current position and generate guidance."""
        info = GuidanceInfo(overlap=overlap, confidence=confidence, dx=dx, dy=dy)

        if confidence < 0.15:
            info.status = ScanStatus.LOST
            info.message = "Tracking lost - move back slowly"
            return info

        if overlap > self.max_overlap:
            info.status = ScanStatus.TOO_MUCH_OVERLAP
            info.message = "Too much overlap - move further"
            info.direction = self._suggest_direction(dx, dy, frame_w, frame_h)
        elif overlap < self.min_overlap:
            info.status = ScanStatus.TOO_LITTLE_OVERLAP
            info.message = "Too little overlap - move back"
            info.direction = self._suggest_back_direction(dx, dy)
        elif self.ideal_overlap_low <= overlap <= self.ideal_overlap_high:
            info.status = ScanStatus.GOOD_OVERLAP
            info.message = f"Good overlap ({overlap:.0%}) - stitching"
            info.direction = self._suggest_direction(dx, dy, frame_w, frame_h)
        else:
            info.status = ScanStatus.TRACKING
            info.message = f"Overlap: {overlap:.0%}"
            info.direction = self._suggest_direction(dx, dy, frame_w, frame_h)

        return info

    def should_stitch(self, overlap: float, confidence: float) -> bool:
        """Determine if the current frame should be stitched."""
        if confidence < 0.2:
            return False
        new_content = 1.0 - overlap
        return (self.min_overlap <= overlap <= self.max_overlap and
                new_content >= self.stitch_threshold)

    def _suggest_direction(self, dx: float, dy: float,
                           frame_w: int, frame_h: int) -> Direction:
        """Suggest direction to continue scanning (move away from panorama center)."""
        if abs(dx) < 5 and abs(dy) < 5:
            return Direction.RIGHT  # Default scanning direction

        # Suggest continuing in the current movement direction
        if abs(dx) > abs(dy) * 1.5:
            return Direction.RIGHT if dx > 0 else Direction.LEFT
        elif abs(dy) > abs(dx) * 1.5:
            return Direction.DOWN if dy > 0 else Direction.UP
        else:
            if dx > 0 and dy > 0:
                return Direction.RIGHT_DOWN
            elif dx > 0 and dy < 0:
                return Direction.RIGHT_UP
            elif dx < 0 and dy > 0:
                return Direction.LEFT_DOWN
            else:
                return Direction.LEFT_UP

    def _suggest_back_direction(self, dx: float, dy: float) -> Direction:
        """Suggest moving back toward the panorama."""
        if abs(dx) > abs(dy):
            return Direction.LEFT if dx > 0 else Direction.RIGHT
        else:
            return Direction.UP if dy > 0 else Direction.DOWN
