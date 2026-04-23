"""
Lightweight frame quality and stitchability analysis.
Used both in preview mode and during live scanning.
"""

import math
from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np

from .stitcher import ImageStitcher, MatchResult


@dataclass
class QualityAssessment:
    """Describes whether the current view is suitable for stitching."""

    focus_score: float = 0.0
    brightness: float = 0.0
    detail_brightness: float = 0.0
    active_ratio: float = 0.0
    contrast: float = 0.0
    texture_points: int = 0
    frame_score: float = 0.0
    stitchability_score: float = 0.0
    match_confidence: float = 0.0
    motion_pixels: float = 0.0
    is_frame_usable: bool = False
    is_pair_stitchable: bool = False
    summary: str = "Awaiting frames"
    match: Optional[MatchResult] = field(default=None, repr=False)


class FrameQualityAnalyzer:
    """Evaluates frame quality and pairwise stitchability."""

    def __init__(self):
        self.detector = cv2.ORB_create(nfeatures=800)
        self.matcher = ImageStitcher(max_features=1200)

        self.min_focus = 60.0
        self.min_contrast = 18.0
        self.min_texture_points = 80
        self.min_match_confidence = 0.18
        self.min_motion_pixels = 4.0
        self.max_motion_pixels = 220.0
        self.min_brightness = 45.0
        self.max_brightness = 210.0
        self.min_detail_brightness = 110.0
        self.min_active_ratio = 0.08

    def analyze(
        self,
        frame: np.ndarray,
        previous_frame: Optional[np.ndarray] = None,
        match: Optional[MatchResult] = None,
    ) -> QualityAssessment:
        """Analyze the current frame, optionally with a previous frame."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame

        focus_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        brightness = float(gray.mean())
        detail_brightness = float(np.percentile(gray, 90))
        active_ratio = float(np.count_nonzero(gray > 24) / gray.size)
        contrast = float(gray.std())
        texture_points = len(self.detector.detect(gray, None))

        focus_norm = self._normalize(focus_score, self.min_focus * 0.5, self.min_focus * 3.0)
        contrast_norm = self._normalize(contrast, self.min_contrast * 0.5, self.min_contrast * 2.2)
        texture_norm = self._normalize(
            float(texture_points),
            self.min_texture_points * 0.4,
            self.min_texture_points * 2.5,
        )
        brightness_usable = self._is_brightness_usable(brightness, detail_brightness, active_ratio)
        brightness_norm = self._brightness_score(brightness, detail_brightness, active_ratio)

        frame_score = (
            0.35 * focus_norm +
            0.30 * texture_norm +
            0.20 * brightness_norm +
            0.15 * contrast_norm
        )
        is_frame_usable = (
            focus_score >= self.min_focus and
            contrast >= self.min_contrast and
            texture_points >= self.min_texture_points and
            brightness_usable
        )

        assessment = QualityAssessment(
            focus_score=focus_score,
            brightness=brightness,
            detail_brightness=detail_brightness,
            active_ratio=active_ratio,
            contrast=contrast,
            texture_points=texture_points,
            frame_score=frame_score,
            stitchability_score=frame_score,
            is_frame_usable=is_frame_usable,
            summary=self._frame_summary(
                focus_score,
                brightness,
                detail_brightness,
                active_ratio,
                contrast,
                texture_points,
                is_frame_usable,
            ),
        )

        if previous_frame is None:
            if is_frame_usable:
                assessment.summary = "Preview ready - move the sample slightly to test stitching"
            return assessment

        if match is None:
            match = self.matcher.find_translation(previous_frame, frame)

        motion_pixels = math.hypot(match.dx, match.dy)
        match_confidence = float(match.confidence)
        match_norm = self._normalize(match_confidence, 0.08, 0.40)
        motion_norm = self._normalize(motion_pixels, self.min_motion_pixels, 40.0)
        pair_score = 0.65 * match_norm + 0.35 * motion_norm

        assessment.match = match
        assessment.match_confidence = match_confidence
        assessment.motion_pixels = motion_pixels
        assessment.stitchability_score = 0.5 * frame_score + 0.5 * pair_score
        assessment.is_pair_stitchable = (
            is_frame_usable and
            self.min_motion_pixels <= motion_pixels <= self.max_motion_pixels and
            match_confidence >= self.min_match_confidence
        )
        assessment.summary = self._pair_summary(
            assessment,
            brightness=brightness,
            contrast=contrast,
            texture_points=texture_points,
        )
        return assessment

    def _frame_summary(
        self,
        focus_score: float,
        brightness: float,
        detail_brightness: float,
        active_ratio: float,
        contrast: float,
        texture_points: int,
        is_frame_usable: bool,
    ) -> str:
        if not self._is_brightness_usable(brightness, detail_brightness, active_ratio):
            if brightness > self.max_brightness:
                return "Frame too bright - reduce light or exposure"
            return "Frame too dark - increase light or exposure"
        if focus_score < self.min_focus:
            return "Frame is blurry - refocus before scanning"
        if contrast < self.min_contrast:
            return "Low contrast - adjust light or sample distance"
        if texture_points < self.min_texture_points:
            return "Not enough detail - move to a more textured area"
        if is_frame_usable:
            return "Frame quality looks good"
        return "Frame quality is unstable"

    def _pair_summary(
        self,
        assessment: QualityAssessment,
        brightness: float,
        contrast: float,
        texture_points: int,
    ) -> str:
        if not assessment.is_frame_usable:
            return self._frame_summary(
                assessment.focus_score,
                brightness,
                assessment.detail_brightness,
                assessment.active_ratio,
                contrast,
                texture_points,
                assessment.is_frame_usable,
            )
        if assessment.motion_pixels < self.min_motion_pixels:
            return "Frame is stable - move the sample slightly to test stitching"
        if assessment.motion_pixels > self.max_motion_pixels:
            return "Movement too large - slow down to keep overlap"
        if assessment.match_confidence < self.min_match_confidence:
            return "Weak frame match - keep 20% to 50% overlap"
        return (
            f"Stitchable view ({assessment.motion_pixels:.1f}px, "
            f"confidence {assessment.match_confidence:.0%})"
        )

    @staticmethod
    def _normalize(value: float, low: float, high: float) -> float:
        if high <= low:
            return 0.0
        return max(0.0, min(1.0, (value - low) / (high - low)))

    def _brightness_score(self, brightness: float, detail_brightness: float, active_ratio: float) -> float:
        if self.min_brightness <= brightness <= self.max_brightness:
            return 1.0

        if self._is_brightness_usable(brightness, detail_brightness, active_ratio):
            usable_ratio = self._normalize(active_ratio, self.min_active_ratio, 0.35)
            detail_ratio = self._normalize(detail_brightness, self.min_detail_brightness, 220.0)
            return 0.65 + 0.2 * usable_ratio + 0.15 * detail_ratio

        if brightness < self.min_brightness:
            return self._normalize(brightness, 0.0, self.min_brightness)

        distance = brightness - self.max_brightness
        return max(0.0, 1.0 - distance / 45.0)

    def _is_brightness_usable(
        self,
        brightness: float,
        detail_brightness: float,
        active_ratio: float,
    ) -> bool:
        if self.min_brightness <= brightness <= self.max_brightness:
            return True

        if brightness > self.max_brightness:
            return False

        return (
            active_ratio >= self.min_active_ratio and
            detail_brightness >= self.min_detail_brightness
        )