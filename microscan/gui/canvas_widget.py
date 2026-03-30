"""
Zoomable, pannable panorama canvas widget.
Displays the stitched panorama with navigation controls.
"""

from PyQt5.QtWidgets import QWidget, QSizePolicy
from PyQt5.QtCore import Qt, QPointF, QRectF
from PyQt5.QtGui import QImage, QPixmap, QPainter, QColor, QFont, QWheelEvent, QMouseEvent
import cv2
import numpy as np
from typing import Optional


class CanvasWidget(QWidget):
    """Zoomable, pannable panorama display widget."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._panorama: Optional[np.ndarray] = None
        self._pixmap: Optional[QPixmap] = None
        self._zoom: float = 1.0
        self._min_zoom: float = 0.05
        self._max_zoom: float = 10.0
        self._pan_offset: QPointF = QPointF(0, 0)
        self._dragging: bool = False
        self._drag_start: QPointF = QPointF()
        self._fit_on_next: bool = True

        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setMinimumSize(320, 240)
        self.setMouseTracking(True)

    def update_panorama(self, panorama: np.ndarray):
        """Update the displayed panorama."""
        if panorama is None:
            return

        self._panorama = panorama
        rgb = cv2.cvtColor(panorama, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qimg = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
        self._pixmap = QPixmap.fromImage(qimg.copy())

        if self._fit_on_next:
            self.fit_view()
            self._fit_on_next = False

        self.update()

    def fit_view(self):
        """Fit panorama to widget."""
        if self._pixmap is None:
            return

        pw = self._pixmap.width()
        ph = self._pixmap.height()
        ww = self.width()
        wh = self.height()

        if pw == 0 or ph == 0:
            return

        scale_x = ww / pw
        scale_y = wh / ph
        self._zoom = min(scale_x, scale_y) * 0.95

        # Center
        self._pan_offset = QPointF(
            (ww - pw * self._zoom) / 2,
            (wh - ph * self._zoom) / 2
        )
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.SmoothPixmapTransform)

        # Background
        painter.fillRect(self.rect(), QColor(30, 30, 30))

        if self._pixmap is not None:
            # Draw panorama with zoom and pan
            painter.translate(self._pan_offset)
            painter.scale(self._zoom, self._zoom)
            painter.drawPixmap(0, 0, self._pixmap)
            painter.resetTransform()

            # Info overlay
            self._draw_info(painter)
        else:
            # Placeholder
            painter.setPen(QColor(120, 120, 120))
            font = QFont("Consolas", 12)
            painter.setFont(font)
            painter.drawText(self.rect(), Qt.AlignCenter,
                             "Panorama will appear here\nStart scanning or load a video")

        painter.end()

    def _draw_info(self, painter: QPainter):
        """Draw info overlay (zoom, size)."""
        if self._panorama is None:
            return

        h, w = self._panorama.shape[:2]
        info_text = f"{w}×{h}  {self._zoom:.0%}"

        painter.fillRect(4, self.height() - 28, len(info_text) * 8 + 12, 24,
                         QColor(0, 0, 0, 160))
        painter.setPen(QColor(200, 200, 200))
        font = QFont("Consolas", 9)
        painter.setFont(font)
        painter.drawText(10, self.height() - 10, info_text)

    def wheelEvent(self, event: QWheelEvent):
        """Zoom with mouse wheel."""
        # Zoom toward cursor position
        cursor_pos = event.pos()
        old_zoom = self._zoom

        factor = 1.15 if event.angleDelta().y() > 0 else 1 / 1.15
        self._zoom = max(self._min_zoom, min(self._max_zoom, self._zoom * factor))

        # Adjust pan to zoom toward cursor
        if old_zoom != 0:
            ratio = self._zoom / old_zoom
            self._pan_offset = QPointF(
                cursor_pos.x() - ratio * (cursor_pos.x() - self._pan_offset.x()),
                cursor_pos.y() - ratio * (cursor_pos.y() - self._pan_offset.y())
            )

        self.update()

    def mousePressEvent(self, event: QMouseEvent):
        if event.button() == Qt.LeftButton:
            self._dragging = True
            self._drag_start = event.pos() - self._pan_offset.toPoint()
            self.setCursor(Qt.ClosedHandCursor)
        elif event.button() == Qt.MiddleButton:
            self.fit_view()

    def mouseReleaseEvent(self, event: QMouseEvent):
        if event.button() == Qt.LeftButton:
            self._dragging = False
            self.setCursor(Qt.ArrowCursor)

    def mouseMoveEvent(self, event: QMouseEvent):
        if self._dragging:
            self._pan_offset = QPointF(event.pos() - self._drag_start)
            self.update()

    def mouseDoubleClickEvent(self, event: QMouseEvent):
        """Double-click to fit view."""
        self.fit_view()

    def reset_view(self):
        """Reset to initial state."""
        self._panorama = None
        self._pixmap = None
        self._zoom = 1.0
        self._pan_offset = QPointF(0, 0)
        self._fit_on_next = True
        self.update()
