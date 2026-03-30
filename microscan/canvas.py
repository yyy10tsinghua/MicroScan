"""
Dynamic panorama canvas that grows as new images are stitched.
Manages coordinate tracking, canvas expansion, and image blending.
"""

import cv2
import numpy as np
from typing import Tuple, Optional
from .stitcher import create_weight_map


class PanoramaCanvas:
    """Manages a dynamically expanding panorama canvas."""

    INITIAL_PAD_FACTOR = 2  # Initial padding multiplier

    def __init__(self, blend_mode: str = 'feather'):
        self.canvas: Optional[np.ndarray] = None
        self.canvas_mask: Optional[np.ndarray] = None
        self.blend_mode = blend_mode  # 'feather' or 'overwrite'
        # Offset from canvas origin to the first image's origin
        self.offset_x: int = 0
        self.offset_y: int = 0
        self.image_count: int = 0

    @property
    def initialized(self) -> bool:
        return self.canvas is not None

    def initialize(self, image: np.ndarray):
        """Initialize canvas with the first image."""
        h, w = image.shape[:2]
        pad_h = h * self.INITIAL_PAD_FACTOR
        pad_w = w * self.INITIAL_PAD_FACTOR

        canvas_h = h + 2 * pad_h
        canvas_w = w + 2 * pad_w

        self.canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
        self.canvas_mask = np.zeros((canvas_h, canvas_w), dtype=np.float32)

        self.offset_x = pad_w
        self.offset_y = pad_h

        # Place first image
        self.canvas[self.offset_y:self.offset_y + h, self.offset_x:self.offset_x + w] = image
        self.canvas_mask[self.offset_y:self.offset_y + h, self.offset_x:self.offset_x + w] = 1.0
        self.image_count = 1

    def add_image(self, image: np.ndarray, dx: float, dy: float) -> bool:
        """Add a new image at offset (dx, dy) from the first image's origin.
        dx, dy are in the coordinate system of the first image."""
        if self.canvas is None:
            self.initialize(image)
            return True

        h, w = image.shape[:2]
        # Target position on canvas
        cx = int(round(self.offset_x + dx))
        cy = int(round(self.offset_y + dy))

        # Expand canvas if needed
        self._expand_if_needed(cx, cy, w, h)

        # Recalculate after potential expansion
        cx = int(round(self.offset_x + dx))
        cy = int(round(self.offset_y + dy))

        # Blend image into canvas
        if self.blend_mode == 'feather':
            self._blend_feather(image, cx, cy)
        else:
            self._blend_overwrite(image, cx, cy)

        self.image_count += 1
        return True

    def _blend_feather(self, image: np.ndarray, cx: int, cy: int):
        """Blend image using distance-based feathering."""
        h, w = image.shape[:2]
        new_weight = create_weight_map(h, w)

        # Clamp to canvas bounds
        src_x1 = max(0, -cx)
        src_y1 = max(0, -cy)
        src_x2 = min(w, self.canvas.shape[1] - cx)
        src_y2 = min(h, self.canvas.shape[0] - cy)

        dst_x1 = max(0, cx)
        dst_y1 = max(0, cy)
        dst_x2 = dst_x1 + (src_x2 - src_x1)
        dst_y2 = dst_y1 + (src_y2 - src_y1)

        if dst_x2 <= dst_x1 or dst_y2 <= dst_y1:
            return

        roi_canvas = self.canvas[dst_y1:dst_y2, dst_x1:dst_x2].astype(np.float32)
        roi_mask = self.canvas_mask[dst_y1:dst_y2, dst_x1:dst_x2]
        roi_image = image[src_y1:src_y2, src_x1:src_x2].astype(np.float32)
        roi_weight = new_weight[src_y1:src_y2, src_x1:src_x2]

        # Weighted blend
        total_weight = roi_mask + roi_weight
        total_weight = np.where(total_weight == 0, 1.0, total_weight)

        blended = (roi_canvas * roi_mask[:, :, np.newaxis] +
                   roi_image * roi_weight[:, :, np.newaxis]) / total_weight[:, :, np.newaxis]

        self.canvas[dst_y1:dst_y2, dst_x1:dst_x2] = np.clip(blended, 0, 255).astype(np.uint8)
        self.canvas_mask[dst_y1:dst_y2, dst_x1:dst_x2] = np.maximum(roi_mask, roi_weight)

    def _blend_overwrite(self, image: np.ndarray, cx: int, cy: int):
        """Simple overwrite blending - new image on top."""
        h, w = image.shape[:2]
        src_x1 = max(0, -cx)
        src_y1 = max(0, -cy)
        src_x2 = min(w, self.canvas.shape[1] - cx)
        src_y2 = min(h, self.canvas.shape[0] - cy)

        dst_x1 = max(0, cx)
        dst_y1 = max(0, cy)
        dst_x2 = dst_x1 + (src_x2 - src_x1)
        dst_y2 = dst_y1 + (src_y2 - src_y1)

        if dst_x2 <= dst_x1 or dst_y2 <= dst_y1:
            return

        self.canvas[dst_y1:dst_y2, dst_x1:dst_x2] = image[src_y1:src_y2, src_x1:src_x2]
        self.canvas_mask[dst_y1:dst_y2, dst_x1:dst_x2] = 1.0

    def _expand_if_needed(self, cx: int, cy: int, w: int, h: int):
        """Expand canvas if the new image would fall outside bounds."""
        expand_top = max(0, -cy)
        expand_left = max(0, -cx)
        expand_bottom = max(0, (cy + h) - self.canvas.shape[0])
        expand_right = max(0, (cx + w) - self.canvas.shape[1])

        # Add extra padding when expanding
        pad = max(h, w)
        if expand_top > 0:
            expand_top += pad
        if expand_left > 0:
            expand_left += pad
        if expand_bottom > 0:
            expand_bottom += pad
        if expand_right > 0:
            expand_right += pad

        if expand_top == 0 and expand_left == 0 and expand_bottom == 0 and expand_right == 0:
            return

        old_h, old_w = self.canvas.shape[:2]
        new_h = old_h + expand_top + expand_bottom
        new_w = old_w + expand_left + expand_right

        new_canvas = np.zeros((new_h, new_w, 3), dtype=np.uint8)
        new_mask = np.zeros((new_h, new_w), dtype=np.float32)

        new_canvas[expand_top:expand_top + old_h, expand_left:expand_left + old_w] = self.canvas
        new_mask[expand_top:expand_top + old_h, expand_left:expand_left + old_w] = self.canvas_mask

        self.canvas = new_canvas
        self.canvas_mask = new_mask
        self.offset_x += expand_left
        self.offset_y += expand_top

    def compute_frame_overlap(self, frame: np.ndarray, dx: float, dy: float) -> float:
        """Compute what fraction of the frame overlaps with existing content."""
        if self.canvas is None:
            return 0.0

        h, w = frame.shape[:2]
        cx = int(round(self.offset_x + dx))
        cy = int(round(self.offset_y + dy))

        # Clamp to canvas bounds
        x1 = max(0, cx)
        y1 = max(0, cy)
        x2 = min(self.canvas.shape[1], cx + w)
        y2 = min(self.canvas.shape[0], cy + h)

        if x2 <= x1 or y2 <= y1:
            return 0.0

        region_mask = self.canvas_mask[y1:y2, x1:x2]
        filled_pixels = np.count_nonzero(region_mask > 0)
        frame_pixels = h * w

        return filled_pixels / frame_pixels if frame_pixels > 0 else 0.0

    def get_panorama(self) -> Optional[np.ndarray]:
        """Return the cropped panorama (content region only)."""
        if self.canvas is None:
            return None

        filled = self.canvas_mask > 0
        rows = np.any(filled, axis=1)
        cols = np.any(filled, axis=0)

        if not rows.any() or not cols.any():
            return None

        y1, y2 = np.where(rows)[0][[0, -1]]
        x1, x2 = np.where(cols)[0][[0, -1]]

        return self.canvas[y1:y2 + 1, x1:x2 + 1].copy()

    def get_panorama_size(self) -> Tuple[int, int]:
        """Return (width, height) of the content region."""
        pano = self.get_panorama()
        if pano is None:
            return (0, 0)
        return (pano.shape[1], pano.shape[0])

    def get_edge_region(self, frame_shape: Tuple[int, int],
                        dx: float, dy: float, margin: int = 50) -> Optional[np.ndarray]:
        """Extract the canvas region near where a new frame would be placed.
        Used for matching the current frame against the panorama."""
        if self.canvas is None:
            return None

        h, w = frame_shape[:2]
        cx = int(round(self.offset_x + dx))
        cy = int(round(self.offset_y + dy))

        # Extract with margin
        x1 = max(0, cx - margin)
        y1 = max(0, cy - margin)
        x2 = min(self.canvas.shape[1], cx + w + margin)
        y2 = min(self.canvas.shape[0], cy + h + margin)

        if x2 <= x1 or y2 <= y1:
            return None

        return self.canvas[y1:y2, x1:x2].copy()

    def reset(self):
        """Reset the canvas to empty state."""
        self.canvas = None
        self.canvas_mask = None
        self.offset_x = 0
        self.offset_y = 0
        self.image_count = 0
