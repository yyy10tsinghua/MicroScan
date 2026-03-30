"""
Video-to-panorama stitcher.
Extracts frames from video, detects overlap, and stitches into a large image.
"""

import cv2
import numpy as np
from typing import Optional, Callable
from .stitcher import ImageStitcher
from .canvas import PanoramaCanvas


class VideoStitcher:
    """Stitch video frames into a panorama."""

    def __init__(self, blend_mode: str = 'feather'):
        self.stitcher = ImageStitcher()
        self.canvas = PanoramaCanvas(blend_mode=blend_mode)
        self.skip_frames: int = 1  # Process every N-th frame
        self.min_movement: float = 15.0  # Min pixels of movement to consider a frame
        self.max_movement: float = 0  # 0 = auto (half frame diagonal)
        self.min_confidence: float = 0.2
        self.min_overlap: float = 0.10
        self.max_overlap: float = 0.85

    def process_video(self, video_path: str,
                      progress_callback: Optional[Callable[[float, str], None]] = None
                      ) -> Optional[np.ndarray]:
        """Process a video file and return stitched panorama.
        
        Args:
            video_path: Path to the video file.
            progress_callback: Optional callback(progress_0_to_1, message).
        
        Returns:
            Stitched panorama image, or None on failure.
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            if progress_callback:
                progress_callback(0, "Failed to open video file")
            return None

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        if progress_callback:
            progress_callback(0, f"Video: {total_frames} frames @ {fps:.1f} FPS")

        self.canvas.reset()
        last_frame = None
        cum_dx, cum_dy = 0.0, 0.0
        last_stitch_dx, last_stitch_dy = 0.0, 0.0
        stitched_count = 0
        frame_idx = 0
        processed = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_idx += 1
            if frame_idx % self.skip_frames != 0:
                continue

            processed += 1

            # Auto-compute max movement from first frame
            if self.max_movement == 0 and frame is not None:
                h, w = frame.shape[:2]
                self.max_movement = np.sqrt(h ** 2 + w ** 2) * 0.5

            # First frame
            if last_frame is None:
                self.canvas.initialize(frame)
                last_frame = frame.copy()
                stitched_count = 1
                if progress_callback:
                    progress_callback(frame_idx / max(total_frames, 1),
                                      f"Initialized with frame 1")
                continue

            # Find translation from last processed frame
            match = self.stitcher.find_translation(last_frame, frame)

            if match.confidence < self.min_confidence:
                # Skip frame - can't match
                if progress_callback and processed % 10 == 0:
                    progress_callback(frame_idx / max(total_frames, 1),
                                      f"Frame {frame_idx}: low confidence, skipping")
                continue

            movement = np.sqrt(match.dx ** 2 + match.dy ** 2)

            # Skip if too little movement (essentially same frame)
            if movement < self.min_movement:
                last_frame = frame.copy()
                cum_dx += match.dx
                cum_dy += match.dy
                continue

            # Skip if too much movement (likely a jump/error)
            if movement > self.max_movement:
                if progress_callback:
                    progress_callback(frame_idx / max(total_frames, 1),
                                      f"Frame {frame_idx}: excessive movement, skipping")
                continue

            cum_dx += match.dx
            cum_dy += match.dy
            last_frame = frame.copy()

            # Check movement since last stitch
            move_since = np.sqrt(
                (cum_dx - last_stitch_dx) ** 2 +
                (cum_dy - last_stitch_dy) ** 2
            )

            if move_since < self.min_movement:
                continue

            # Check overlap
            overlap = self.canvas.compute_frame_overlap(frame, cum_dx, cum_dy)

            if self.min_overlap <= overlap <= self.max_overlap:
                self.canvas.add_image(frame, cum_dx, cum_dy)
                last_stitch_dx = cum_dx
                last_stitch_dy = cum_dy
                stitched_count += 1
                if progress_callback:
                    progress_callback(
                        frame_idx / max(total_frames, 1),
                        f"Stitched frame {frame_idx} ({stitched_count} total, "
                        f"overlap {overlap:.0%})"
                    )

        cap.release()

        if progress_callback:
            pano = self.canvas.get_panorama()
            if pano is not None:
                progress_callback(1.0,
                                  f"Done! {stitched_count} frames stitched, "
                                  f"panorama: {pano.shape[1]}x{pano.shape[0]}")
            else:
                progress_callback(1.0, "Failed to create panorama")

        return self.canvas.get_panorama()
