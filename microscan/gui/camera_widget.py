"""
Camera preview widget with overlay for guidance information.
"""

from typing import Optional

from PyQt5.QtWidgets import QWidget, QSizePolicy
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap, QPainter, QColor, QFont, QPen, QPaintEvent
import cv2
import numpy as np
from ..overlap import GuidanceInfo, ScanStatus, Direction


class CameraWidget(QWidget):
    """Widget displaying live camera feed with scanning overlay."""

    frame_ready = pyqtSignal(np.ndarray)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._frame: Optional[np.ndarray] = None
        self._guidance: GuidanceInfo = GuidanceInfo()
        self._show_overlay: bool = True
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setMinimumSize(320, 240)

    def update_frame(self, frame: np.ndarray):
        """Update displayed frame."""
        self._frame = frame.copy()
        self.update()

    def update_guidance(self, guidance: GuidanceInfo):
        """Update overlay guidance."""
        self._guidance = guidance
        self.update()

    def clear(self):
        """Clear the preview frame and overlay."""
        self._frame = None
        self._guidance = GuidanceInfo()
        self.update()

    def set_overlay_visible(self, visible: bool):
        self._show_overlay = visible
        self.update()

    def paintEvent(self, a0: Optional[QPaintEvent]):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.SmoothPixmapTransform)
        painter.fillRect(self.rect(), QColor(22, 22, 22))

        if self._frame is None:
            painter.setPen(QColor(120, 120, 120))
            font = QFont("Consolas", 12)
            painter.setFont(font)
            painter.drawText(
                self.rect(),
                Qt.AlignmentFlag.AlignCenter,
                "Connect a camera to start preview\nOverlay will show stitchability hints"
            )
            painter.end()
            return

        # Convert frame to QPixmap
        frame_rgb = cv2.cvtColor(self._frame, cv2.COLOR_BGR2RGB)
        h, w, ch = frame_rgb.shape
        qimg = QImage(frame_rgb.data, w, h, ch * w, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg)

        # Scale to widget keeping aspect ratio
        scaled = pixmap.scaled(
            self.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        x = (self.width() - scaled.width()) // 2
        y = (self.height() - scaled.height()) // 2
        painter.drawPixmap(x, y, scaled)

        # Draw overlay
        if self._show_overlay and self._guidance.status != ScanStatus.IDLE:
            self._draw_overlay(painter, x, y, scaled.width(), scaled.height())

        painter.end()

    def _draw_overlay(self, painter: QPainter, x: int, y: int, w: int, h: int):
        """Draw scanning guidance overlay."""
        g = self._guidance

        # Border color based on status
        b, gr, r = g.color
        border_color = QColor(r, gr, b)
        pen = QPen(border_color, 4)
        painter.setPen(pen)
        painter.drawRect(x + 2, y + 2, w - 4, h - 4)

        # Status bar at bottom
        bar_h = 36
        bar_rect = (x, y + h - bar_h, w, bar_h)

        painter.fillRect(*bar_rect, QColor(0, 0, 0, 180))
        painter.setPen(QColor(255, 255, 255))
        font = QFont("Consolas", 10)
        font.setBold(True)
        painter.setFont(font)

        # Status text
        status_text = g.message or g.status.value
        painter.drawText(x + 8, y + h - bar_h + 4, w - 16, bar_h - 8,
                         Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter, status_text)

        # Overlap percentage bar (top-right)
        if g.overlap > 0:
            self._draw_overlap_bar(painter, x + w - 120, y + 8, 110, 20, g.overlap)

        # Direction arrow
        if g.direction != Direction.NONE:
            self._draw_direction_arrow(painter, x + w // 2, y + h // 2,
                                       g.direction, border_color)

    def _draw_overlap_bar(self, painter: QPainter, x: int, y: int,
                          w: int, h: int, overlap: float):
        """Draw overlap percentage bar."""
        painter.fillRect(x, y, w, h, QColor(0, 0, 0, 160))
        # Fill bar
        fill_w = int(w * min(overlap, 1.0))
        if overlap < 0.15:
            color = QColor(200, 50, 50)
        elif overlap > 0.70:
            color = QColor(200, 150, 50)
        else:
            color = QColor(50, 200, 50)
        painter.fillRect(x, y, fill_w, h, color)
        # Text
        painter.setPen(QColor(255, 255, 255))
        font = QFont("Consolas", 8)
        painter.setFont(font)
        painter.drawText(x, y, w, h, Qt.AlignmentFlag.AlignCenter, f"{overlap:.0%}")

    def _draw_direction_arrow(self, painter: QPainter, cx: int, cy: int,
                              direction: Direction, color: QColor):
        """Draw a directional guidance arrow."""
        arrow_len = 50
        painter.setPen(QPen(color, 3))

        dx_map = {
            Direction.LEFT: (-1, 0), Direction.RIGHT: (1, 0),
            Direction.UP: (0, -1), Direction.DOWN: (0, 1),
            Direction.LEFT_UP: (-0.7, -0.7), Direction.LEFT_DOWN: (-0.7, 0.7),
            Direction.RIGHT_UP: (0.7, -0.7), Direction.RIGHT_DOWN: (0.7, 0.7),
        }

        if direction in dx_map:
            adx, ady = dx_map[direction]
            ex = int(cx + adx * arrow_len)
            ey = int(cy + ady * arrow_len)
            painter.drawLine(cx, cy, ex, ey)
            # Arrowhead
            import math
            angle = math.atan2(ady, adx)
            head_len = 15
            for offset in [2.5, -2.5]:
                hx = int(ex - head_len * math.cos(angle + offset))
                hy = int(ey - head_len * math.sin(angle + offset))
                painter.drawLine(ex, ey, hx, hy)
