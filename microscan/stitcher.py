"""
Core image stitching engine.
Uses feature-based registration (ORB) with phase correlation fallback.
Optimized for microscopy: mostly translational movement, high texture.
"""

import cv2
import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class MatchResult:
    """Result of image matching."""
    dx: float = 0.0
    dy: float = 0.0
    confidence: float = 0.0
    num_matches: int = 0
    homography: Optional[np.ndarray] = None


class ImageStitcher:
    """Feature-based image stitching engine."""

    def __init__(self, max_features: int = 3000, ratio_threshold: float = 0.75):
        self.max_features = max_features
        self.ratio_threshold = ratio_threshold
        self.detector = cv2.ORB_create(nfeatures=max_features)
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING)

    def find_translation(self, img1: np.ndarray, img2: np.ndarray) -> MatchResult:
        """Find translation between two images. Tries feature matching, falls back to phase correlation."""
        result = self._match_features(img1, img2)
        if result.confidence > 0.3:
            return result

        # Fallback: phase correlation
        return self._phase_correlate(img1, img2)

    def _match_features(self, img1: np.ndarray, img2: np.ndarray) -> MatchResult:
        """Feature-based translation detection using ORB."""
        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY) if len(img1.shape) == 3 else img1
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY) if len(img2.shape) == 3 else img2

        kp1, desc1 = self.detector.detectAndCompute(gray1, None)
        kp2, desc2 = self.detector.detectAndCompute(gray2, None)

        if desc1 is None or desc2 is None or len(desc1) < 2 or len(desc2) < 2:
            return MatchResult()

        raw_matches = self.matcher.knnMatch(desc1, desc2, k=2)
        good = []
        for pair in raw_matches:
            if len(pair) == 2:
                m, n = pair
                if m.distance < self.ratio_threshold * n.distance:
                    good.append(m)

        if len(good) < 8:
            return MatchResult(num_matches=len(good))

        src_pts = np.float32([kp1[m.queryIdx].pt for m in good])
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good])

        # Compute robust translation using RANSAC-filtered median
        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        if H is None:
            return MatchResult(num_matches=len(good))

        inlier_mask = mask.ravel().astype(bool)
        inlier_src = src_pts[inlier_mask]
        inlier_dst = dst_pts[inlier_mask]
        num_inliers = int(inlier_mask.sum())

        if num_inliers < 6:
            return MatchResult(num_matches=num_inliers)

        translations = inlier_dst - inlier_src
        dx = float(np.median(translations[:, 0]))
        dy = float(np.median(translations[:, 1]))

        # Confidence: based on inlier ratio and consistency
        inlier_ratio = num_inliers / len(good)
        std = np.std(translations, axis=0)
        consistency = 1.0 / (1.0 + float(np.mean(std)))
        confidence = min(1.0, inlier_ratio * consistency * 2.0)

        return MatchResult(
            dx=dx, dy=dy,
            confidence=confidence,
            num_matches=num_inliers,
            homography=H
        )

    def _phase_correlate(self, img1: np.ndarray, img2: np.ndarray) -> MatchResult:
        """Phase correlation for sub-pixel translation detection."""
        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY) if len(img1.shape) == 3 else img1
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY) if len(img2.shape) == 3 else img2

        # Ensure same size
        h = min(gray1.shape[0], gray2.shape[0])
        w = min(gray1.shape[1], gray2.shape[1])
        g1 = gray1[:h, :w].astype(np.float64)
        g2 = gray2[:h, :w].astype(np.float64)

        # Apply Hanning window to reduce edge effects
        hann = cv2.createHanningWindow((w, h), cv2.CV_64F)
        g1 = g1 * hann
        g2 = g2 * hann

        (dx, dy), response = cv2.phaseCorrelate(g1, g2)
        confidence = min(1.0, max(0.0, float(response)))

        return MatchResult(dx=dx, dy=dy, confidence=confidence)

    def compute_homography(self, img1: np.ndarray, img2: np.ndarray) -> MatchResult:
        """Compute full homography between two images."""
        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY) if len(img1.shape) == 3 else img1
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY) if len(img2.shape) == 3 else img2

        kp1, desc1 = self.detector.detectAndCompute(gray1, None)
        kp2, desc2 = self.detector.detectAndCompute(gray2, None)

        if desc1 is None or desc2 is None or len(desc1) < 2 or len(desc2) < 2:
            return MatchResult()

        raw_matches = self.matcher.knnMatch(desc1, desc2, k=2)
        good = []
        for pair in raw_matches:
            if len(pair) == 2:
                m, n = pair
                if m.distance < self.ratio_threshold * n.distance:
                    good.append(m)

        if len(good) < 10:
            return MatchResult(num_matches=len(good))

        src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        if H is None:
            return MatchResult(num_matches=len(good))

        num_inliers = int(mask.ravel().sum())
        inlier_ratio = num_inliers / len(good)
        confidence = min(1.0, inlier_ratio * 1.5)

        # Extract translation from homography
        dx = float(H[0, 2])
        dy = float(H[1, 2])

        return MatchResult(
            dx=dx, dy=dy,
            confidence=confidence,
            num_matches=num_inliers,
            homography=H
        )


def create_weight_map(h: int, w: int) -> np.ndarray:
    """Create a distance-based weight map for blending (feathering).
    Pixels near center have weight ~1, pixels near edges have weight ~0."""
    x = np.minimum(np.arange(w, dtype=np.float32), np.arange(w - 1, -1, -1, dtype=np.float32))
    y = np.minimum(np.arange(h, dtype=np.float32), np.arange(h - 1, -1, -1, dtype=np.float32))
    weight = np.minimum(x[np.newaxis, :], y[:, np.newaxis])
    max_val = weight.max()
    if max_val > 0:
        weight /= max_val
    return weight
