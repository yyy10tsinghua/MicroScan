"""
Main application window for MicroScan.
Provides camera scanning mode and video-to-panorama mode.
"""

import os
import cv2
import numpy as np
from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
    QLabel, QComboBox, QFileDialog, QSplitter, QGroupBox, QGridLayout,
    QProgressBar, QStatusBar, QTabWidget, QSpinBox, QCheckBox,
    QSlider, QMessageBox, QApplication
)
from typing import Optional

from PyQt5.QtCore import Qt, QTimer, QThread, pyqtSignal
from PyQt5.QtGui import QFont, QIcon, QCloseEvent

from .camera_widget import CameraWidget
from .canvas_widget import CanvasWidget
from ..camera import MicroscopeCamera
from ..scanner import MicroscopeScanner
from ..video_stitcher import VideoStitcher
from ..overlap import GuidanceInfo, ScanStatus
from ..quality import FrameQualityAnalyzer


class CameraDetectWorker(QThread):
    """Background thread for camera detection (avoids blocking GUI)."""
    finished = pyqtSignal(list)

    def run(self):
        cameras = MicroscopeCamera.list_cameras()
        self.finished.emit(cameras)


class VideoWorker(QThread):
    """Background thread for video processing."""
    progress = pyqtSignal(float, str)
    finished = pyqtSignal(object)  # np.ndarray or None

    def __init__(self, video_path: str, skip_frames: int = 2):
        super().__init__()
        self.video_path = video_path
        self.skip_frames = skip_frames
        self._cancel = False

    def run(self):
        vs = VideoStitcher()
        vs.skip_frames = self.skip_frames

        def on_progress(p, msg):
            if self._cancel:
                return
            self.progress.emit(p, msg)

        result = vs.process_video(self.video_path, on_progress)
        if not self._cancel:
            self.finished.emit(result)

    def cancel(self):
        self._cancel = True


class MainWindow(QMainWindow):
    """MicroScan main application window."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("MicroScan - Microscope Image Stitcher")
        self.setMinimumSize(1100, 700)
        self.resize(1400, 850)

        # Core components
        self.camera = MicroscopeCamera()
        self.scanner = MicroscopeScanner()
        self.preview_analyzer = FrameQualityAnalyzer()
        self._preview_reference_frame = None
        self._preview_frame_counter = 0
        self._last_preview_assessment = None

        # Timers
        self.camera_timer = QTimer()
        self.camera_timer.timeout.connect(self._on_camera_frame)
        self.pano_update_timer = QTimer()
        self.pano_update_timer.timeout.connect(self._update_panorama_view)

        # Video worker
        self.video_worker = None
        self._camera_detect_worker = None

        self._build_ui()
        self._reset_live_feedback(clear_preview=True)
        QTimer.singleShot(0, self._detect_cameras)

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)
        main_layout.setContentsMargins(6, 6, 6, 6)

        # Top: Mode tabs
        self.tabs = QTabWidget()
        main_layout.addWidget(self.tabs)

        # Tab 1: Live Scan
        self.tabs.addTab(self._build_live_tab(), "📷 Live Scan")
        # Tab 2: Video Import
        self.tabs.addTab(self._build_video_tab(), "🎬 Video Import")

        # Status bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready")

    # ── Live Scan Tab ──────────────────────────────────────────────

    def _build_live_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)

        # Controls bar
        controls = QHBoxLayout()

        # Camera selection
        controls.addWidget(QLabel("Camera:"))
        self.camera_combo = QComboBox()
        self.camera_combo.setEditable(True)
        self.camera_combo.setMinimumWidth(160)
        self.camera_combo.addItem("Click Refresh to detect", None)
        controls.addWidget(self.camera_combo)

        self.btn_refresh_cam = QPushButton("🔍 Refresh")
        self.btn_refresh_cam.setToolTip("Detect connected cameras")
        self.btn_refresh_cam.clicked.connect(self._detect_cameras)
        controls.addWidget(self.btn_refresh_cam)

        self.btn_connect = QPushButton("Connect")
        self.btn_connect.clicked.connect(self._toggle_camera)
        controls.addWidget(self.btn_connect)

        controls.addSpacing(20)

        # Resolution
        controls.addWidget(QLabel("Res:"))
        self.res_combo = QComboBox()
        self.res_combo.addItems(["640x480", "800x600", "1280x720", "1920x1080"])
        self.res_combo.setCurrentIndex(2)
        self.res_combo.currentIndexChanged.connect(self._change_resolution)
        controls.addWidget(self.res_combo)

        controls.addSpacing(20)

        # Scan controls
        self.btn_start_scan = QPushButton("▶ Start Scan")
        self.btn_start_scan.setStyleSheet("font-weight: bold; padding: 5px 15px;")
        self.btn_start_scan.clicked.connect(self._toggle_scanning)
        self.btn_start_scan.setEnabled(False)
        controls.addWidget(self.btn_start_scan)

        self.btn_force_stitch = QPushButton("⊕ Force Stitch")
        self.btn_force_stitch.clicked.connect(self._force_stitch)
        self.btn_force_stitch.setEnabled(False)
        controls.addWidget(self.btn_force_stitch)

        controls.addStretch()

        # Save
        self.btn_save = QPushButton("💾 Save Panorama")
        self.btn_save.clicked.connect(self._save_panorama)
        self.btn_save.setEnabled(False)
        controls.addWidget(self.btn_save)

        layout.addLayout(controls)

        # Main content: Camera + Panorama
        splitter = QSplitter(Qt.Orientation.Horizontal)

        # Left: Camera view
        left_group = QGroupBox("Camera Preview")
        left_layout = QVBoxLayout(left_group)
        self.camera_widget = CameraWidget()
        left_layout.addWidget(self.camera_widget)

        # Scan info
        self.lbl_scan_info = QLabel("Images: 0 | Panorama: —")
        self.lbl_scan_info.setFont(QFont("Consolas", 9))
        left_layout.addWidget(self.lbl_scan_info)

        self.live_inspector = self._build_live_inspector()
        left_layout.addWidget(self.live_inspector)

        splitter.addWidget(left_group)

        # Right: Panorama canvas
        right_group = QGroupBox("Panorama")
        right_layout = QVBoxLayout(right_group)
        self.canvas_widget = CanvasWidget()
        right_layout.addWidget(self.canvas_widget)

        pano_controls = QHBoxLayout()
        btn_fit = QPushButton("Fit View")
        btn_fit.clicked.connect(self.canvas_widget.fit_view)
        pano_controls.addWidget(btn_fit)
        pano_controls.addStretch()
        right_layout.addLayout(pano_controls)

        splitter.addWidget(right_group)
        splitter.setSizes([500, 700])

        layout.addWidget(splitter)
        return tab

    def _build_live_inspector(self) -> QGroupBox:
        group = QGroupBox("Stitchability Inspector")
        layout = QGridLayout(group)
        layout.setHorizontalSpacing(14)
        layout.setVerticalSpacing(6)

        labels = [
            ("Camera", "lbl_camera_state_value"),
            ("Mode", "lbl_scan_state_value"),
            ("Verdict", "lbl_stitchability_value"),
            ("Focus", "lbl_focus_value"),
            ("Texture", "lbl_texture_value"),
            ("Brightness", "lbl_brightness_value"),
            ("Motion", "lbl_motion_value"),
            ("Match", "lbl_match_value"),
        ]

        for row, (title, attr_name) in enumerate(labels):
            title_label = QLabel(f"{title}:")
            value_label = QLabel("—")
            value_label.setFont(QFont("Consolas", 9))
            layout.addWidget(title_label, row, 0)
            layout.addWidget(value_label, row, 1)
            setattr(self, attr_name, value_label)

        return group

    def _reset_live_feedback(self, clear_preview: bool = False):
        self._preview_reference_frame = None
        self._preview_frame_counter = 0
        self._last_preview_assessment = None
        self.lbl_scan_info.setText("Images: 0 | Panorama: —")
        self.lbl_camera_state_value.setText("Disconnected")
        self.lbl_scan_state_value.setText("Idle")
        self.lbl_stitchability_value.setText("Awaiting camera")
        self.lbl_stitchability_value.setStyleSheet("color: #dcdcdc;")
        self.lbl_focus_value.setText("—")
        self.lbl_texture_value.setText("—")
        self.lbl_brightness_value.setText("—")
        self.lbl_motion_value.setText("—")
        self.lbl_match_value.setText("—")
        if clear_preview:
            self.camera_widget.clear()

    def _evaluate_preview_frame(self, frame: np.ndarray):
        self._preview_frame_counter += 1
        should_run = (
            self._last_preview_assessment is None or
            self._preview_frame_counter % 3 == 0
        )

        if should_run:
            assessment = self.preview_analyzer.analyze(frame, self._preview_reference_frame)
            if assessment.is_frame_usable and (
                self._preview_reference_frame is None or
                assessment.is_pair_stitchable or
                assessment.match_confidence >= 0.12 or
                assessment.motion_pixels > self.preview_analyzer.max_motion_pixels
            ):
                self._preview_reference_frame = frame.copy()
            self._last_preview_assessment = assessment
        else:
            assessment = self._last_preview_assessment

        return assessment, self._preview_guidance_from_assessment(assessment)

    def _preview_guidance_from_assessment(self, assessment) -> GuidanceInfo:
        if assessment is None:
            return GuidanceInfo(status=ScanStatus.IDLE, message="Awaiting preview")

        if assessment.is_pair_stitchable:
            status = ScanStatus.PREVIEW_READY
            confidence = assessment.match_confidence
        elif assessment.is_frame_usable:
            status = ScanStatus.PREVIEW_WARNING
            confidence = assessment.match_confidence or assessment.frame_score
        else:
            status = ScanStatus.POOR_QUALITY
            confidence = assessment.frame_score

        return GuidanceInfo(
            status=status,
            message=assessment.summary,
            confidence=confidence
        )

    def _update_live_feedback_panel(self, assessment, guidance: GuidanceInfo):
        if self.camera.is_open:
            self.lbl_camera_state_value.setText(
                f"Cam {self.camera.device_id} ({self.camera.backend_name})"
            )
        else:
            self.lbl_camera_state_value.setText("Disconnected")

        self.lbl_scan_state_value.setText("Scanning" if self.scanner.is_scanning else "Preview")

        verdict_text = guidance.message or (assessment.summary if assessment else "Awaiting frames")
        self.lbl_stitchability_value.setText(verdict_text)
        if assessment is not None and assessment.is_pair_stitchable:
            verdict_color = "#67d78a"
        elif assessment is not None and assessment.is_frame_usable:
            verdict_color = "#f0c25b"
        else:
            verdict_color = "#ef7f7f"
        self.lbl_stitchability_value.setStyleSheet(f"color: {verdict_color}; font-weight: bold;")

        if assessment is None:
            self.lbl_focus_value.setText("—")
            self.lbl_texture_value.setText("—")
            self.lbl_brightness_value.setText("—")
            self.lbl_motion_value.setText("—")
            self.lbl_match_value.setText("—")
            return

        self.lbl_focus_value.setText(f"{assessment.focus_score:.0f}")
        self.lbl_texture_value.setText(str(assessment.texture_points))
        self.lbl_brightness_value.setText(f"{assessment.brightness:.0f}")
        self.lbl_motion_value.setText(
            f"{assessment.motion_pixels:.1f}px" if assessment.motion_pixels > 0 else "—"
        )
        self.lbl_match_value.setText(
            f"{assessment.match_confidence:.0%}" if assessment.match_confidence > 0 else "—"
        )

    def _update_scan_info_label(self, assessment, guidance: GuidanceInfo):
        pano_w, pano_h = self.scanner.get_panorama_size()
        pano_text = f"{pano_w}×{pano_h}" if pano_w and pano_h else "—"

        if self.scanner.is_scanning:
            self.lbl_scan_info.setText(
                f"Images: {self.scanner.image_count} | "
                f"Panorama: {pano_text} | "
                f"Overlap: {guidance.overlap:.0%} | "
                f"Conf: {guidance.confidence:.0%}"
            )
            return

        if assessment is None:
            self.lbl_scan_info.setText("Images: 0 | Panorama: —")
            return

        preview_conf = assessment.match_confidence if assessment.match_confidence > 0 else assessment.frame_score
        self.lbl_scan_info.setText(
            f"Preview | Focus: {assessment.focus_score:.0f} | "
            f"Texture: {assessment.texture_points} | "
            f"Confidence: {preview_conf:.0%}"
        )

    # ── Video Import Tab ───────────────────────────────────────────

    def _build_video_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)

        # Controls
        controls = QHBoxLayout()

        self.btn_load_video = QPushButton("📂 Load Video")
        self.btn_load_video.clicked.connect(self._load_video)
        controls.addWidget(self.btn_load_video)

        self.lbl_video_path = QLabel("No video loaded")
        self.lbl_video_path.setStyleSheet("color: gray;")
        controls.addWidget(self.lbl_video_path, 1)

        controls.addWidget(QLabel("Skip frames:"))
        self.spin_skip = QSpinBox()
        self.spin_skip.setRange(1, 30)
        self.spin_skip.setValue(2)
        controls.addWidget(self.spin_skip)

        self.btn_process_video = QPushButton("▶ Process Video")
        self.btn_process_video.setStyleSheet("font-weight: bold; padding: 5px 15px;")
        self.btn_process_video.clicked.connect(self._process_video)
        self.btn_process_video.setEnabled(False)
        controls.addWidget(self.btn_process_video)

        self.btn_cancel_video = QPushButton("✖ Cancel")
        self.btn_cancel_video.clicked.connect(self._cancel_video)
        self.btn_cancel_video.setEnabled(False)
        controls.addWidget(self.btn_cancel_video)

        controls.addSpacing(20)

        self.btn_save_video_pano = QPushButton("💾 Save Panorama")
        self.btn_save_video_pano.clicked.connect(self._save_video_panorama)
        self.btn_save_video_pano.setEnabled(False)
        controls.addWidget(self.btn_save_video_pano)

        layout.addLayout(controls)

        # Progress
        self.video_progress = QProgressBar()
        self.video_progress.setTextVisible(True)
        self.video_progress.setValue(0)
        layout.addWidget(self.video_progress)

        self.lbl_video_status = QLabel("")
        self.lbl_video_status.setFont(QFont("Consolas", 9))
        layout.addWidget(self.lbl_video_status)

        # Panorama view
        self.video_canvas = CanvasWidget()
        layout.addWidget(self.video_canvas, 1)

        return tab

    # ── Camera Management ──────────────────────────────────────────

    def _detect_cameras(self):
        """Detect cameras in a background thread to avoid freezing the GUI."""
        if self._camera_detect_worker and self._camera_detect_worker.isRunning():
            return
        self.btn_refresh_cam.setEnabled(False)
        self.btn_refresh_cam.setText("Detecting...")
        self.status_bar.showMessage("Detecting cameras...")
        self._camera_detect_worker = CameraDetectWorker()
        self._camera_detect_worker.finished.connect(self._on_cameras_detected)
        self._camera_detect_worker.start()

    def _on_cameras_detected(self, cameras: list):
        """Handle camera detection results."""
        self.btn_refresh_cam.setEnabled(True)
        self.btn_refresh_cam.setText("🔍 Refresh")
        self.camera_combo.clear()
        if cameras:
            for cam_id in cameras:
                self.camera_combo.addItem(f"Camera {cam_id}", cam_id)
            self.status_bar.showMessage(f"Found {len(cameras)} camera(s)")
        else:
            self.camera_combo.addItem("No cameras found", None)
            self.status_bar.showMessage("No cameras detected — you can type a device ID manually")

    def _toggle_camera(self):
        if self.camera.is_open:
            self._disconnect_camera()
        else:
            self._connect_camera()

    def _connect_camera(self):
        idx = self.camera_combo.currentIndex()
        if idx < 0:
            return

        device_id = self.camera_combo.currentData()
        if device_id is None:
            # Try parsing the text as a device ID (user typed manually)
            text = self.camera_combo.currentText().strip()
            try:
                device_id = int(text)
            except ValueError:
                self.status_bar.showMessage("Please select a camera or type a device number (e.g. 0)")
                return

        self.status_bar.showMessage(f"Connecting to Camera {device_id}...")
        QApplication.processEvents()  # Show the message immediately

        # Open camera directly with the desired resolution (avoid double-open)
        res_text = self.res_combo.currentText()
        w, h = map(int, res_text.split('x'))

        if self.camera.open(device_id, w, h):

            actual_w, actual_h = self.camera.get_resolution()
            self._reset_live_feedback(clear_preview=False)
            self.btn_connect.setText("Disconnect")
            self.btn_start_scan.setEnabled(True)
            self.camera_timer.start(33)  # ~30 FPS
            self.status_bar.showMessage(
                f"Connected to Camera {device_id} via {self.camera.backend_name} "
                f"({actual_w}×{actual_h})"
            )
        else:
            self.status_bar.showMessage(
                "Failed to connect to camera — try a different device ID"
            )

    def _disconnect_camera(self):
        if self.scanner.is_scanning:
            self._toggle_scanning()

        self.camera_timer.stop()
        self.pano_update_timer.stop()
        self.camera.close()
        self._reset_live_feedback(clear_preview=True)
        self.btn_connect.setText("Connect")
        self.btn_start_scan.setEnabled(False)
        self.btn_force_stitch.setEnabled(False)
        self.status_bar.showMessage("Disconnected")

    def _change_resolution(self):
        if not self.camera.is_open:
            return
        res_text = self.res_combo.currentText()
        w, h = map(int, res_text.split('x'))
        if self.camera.set_resolution(w, h):
            self._preview_reference_frame = None
            self._last_preview_assessment = None
            actual_w, actual_h = self.camera.get_resolution()
            self.status_bar.showMessage(
                f"Camera resolution changed to {actual_w}×{actual_h}"
            )

    # ── Scanning ───────────────────────────────────────────────────

    def _toggle_scanning(self):
        if self.scanner.is_scanning:
            self.scanner.stop_scanning()
            self.pano_update_timer.stop()
            self.btn_start_scan.setText("▶ Start Scan")
            self.btn_start_scan.setStyleSheet("font-weight: bold; padding: 5px 15px;")
            self.btn_force_stitch.setEnabled(False)
            self.btn_save.setEnabled(self.scanner.image_count > 0)
            self.status_bar.showMessage("Scanning stopped")
        else:
            self.scanner.start_scanning()
            self._last_preview_assessment = None
            self.canvas_widget.reset_view()
            self.pano_update_timer.start(500)  # Update panorama every 500ms
            self.btn_start_scan.setText("⏹ Stop Scan")
            self.btn_start_scan.setStyleSheet(
                "font-weight: bold; padding: 5px 15px; background-color: #cc3333; color: white;")
            self.btn_force_stitch.setEnabled(True)
            self.btn_save.setEnabled(False)
            self.status_bar.showMessage("Scanning started - move the sample slowly")

    def _force_stitch(self):
        frame = self.camera.read_frame()
        if frame is not None:
            self.scanner.force_stitch(frame)
            self._update_panorama_view()

    def _on_camera_frame(self):
        """Called by timer to process camera frames."""
        if not self.camera.is_open:
            return
        frame = self.camera.read_frame()
        if frame is None:
            return

        self.camera_widget.update_frame(frame)

        if self.scanner.is_scanning:
            guidance = self.scanner.process_frame(frame)
            assessment = self.scanner.last_quality
        else:
            assessment, guidance = self._evaluate_preview_frame(frame)

        self.camera_widget.update_guidance(guidance)
        self._update_live_feedback_panel(assessment, guidance)
        self._update_scan_info_label(assessment, guidance)

    def _update_panorama_view(self):
        """Periodically update the panorama display."""
        pano = self.scanner.get_panorama()
        if pano is not None:
            self.canvas_widget.update_panorama(pano)

    # ── Video Processing ───────────────────────────────────────────

    def _load_video(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Load Video",
            "",
            "Video Files (*.mp4 *.avi *.mov *.mkv *.wmv);;All Files (*)"
        )
        if path:
            self._video_path = path
            self.lbl_video_path.setText(os.path.basename(path))
            self.lbl_video_path.setStyleSheet("color: white;")
            self.btn_process_video.setEnabled(True)
            self.video_progress.setValue(0)
            self.lbl_video_status.setText("")

    def _process_video(self):
        if not hasattr(self, '_video_path'):
            return

        self.btn_process_video.setEnabled(False)
        self.btn_cancel_video.setEnabled(True)
        self.btn_load_video.setEnabled(False)
        self.btn_save_video_pano.setEnabled(False)
        self.video_canvas.reset_view()

        self.video_worker = VideoWorker(
            self._video_path,
            skip_frames=self.spin_skip.value()
        )
        self.video_worker.progress.connect(self._on_video_progress)
        self.video_worker.finished.connect(self._on_video_finished)
        self.video_worker.start()

    def _cancel_video(self):
        if self.video_worker:
            self.video_worker.cancel()
            self.video_worker.wait()
            self.btn_process_video.setEnabled(True)
            self.btn_cancel_video.setEnabled(False)
            self.btn_load_video.setEnabled(True)
            self.lbl_video_status.setText("Cancelled")

    def _on_video_progress(self, progress: float, message: str):
        self.video_progress.setValue(int(progress * 100))
        self.lbl_video_status.setText(message)

    def _on_video_finished(self, panorama):
        self.btn_process_video.setEnabled(True)
        self.btn_cancel_video.setEnabled(False)
        self.btn_load_video.setEnabled(True)

        if panorama is not None:
            self._video_panorama = panorama
            self.video_canvas.update_panorama(panorama)
            self.btn_save_video_pano.setEnabled(True)
            self.video_progress.setValue(100)
            h, w = panorama.shape[:2]
            self.status_bar.showMessage(f"Video panorama complete: {w}×{h}")
        else:
            self.lbl_video_status.setText("Failed to create panorama from video")
            self.status_bar.showMessage("Video processing failed")

    # ── Save ───────────────────────────────────────────────────────

    def _save_panorama(self):
        pano = self.scanner.get_panorama()
        if pano is None:
            return
        self._do_save(pano)

    def _save_video_panorama(self):
        if hasattr(self, '_video_panorama') and self._video_panorama is not None:
            self._do_save(self._video_panorama)

    def _do_save(self, image: np.ndarray):
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Panorama",
            "panorama.png",
            "PNG (*.png);;JPEG (*.jpg);;TIFF (*.tiff);;BMP (*.bmp)"
        )
        if path:
            cv2.imwrite(path, image)
            self.status_bar.showMessage(f"Saved: {path}")

    # ── Cleanup ────────────────────────────────────────────────────

    def closeEvent(self, a0: Optional[QCloseEvent]):
        self.camera_timer.stop()
        self.pano_update_timer.stop()
        self.camera.close()
        if self.video_worker and self.video_worker.isRunning():
            self.video_worker.cancel()
            self.video_worker.wait()
        if a0 is not None:
            a0.accept()
