import os
import cv2
import json
import time
import numpy as np
import subprocess
from typing import Optional

from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QSplitter,
    QGroupBox, QLabel, QPushButton, QComboBox, QCheckBox,
    QSlider, QFileDialog, QMessageBox, QStatusBar, QMenu,
    QScrollArea, QGridLayout, QFormLayout, QFrame, QApplication,
    QDialog, QRadioButton, QLineEdit
)

# Valid image extensions for cv2.imwrite
_VALID_IMAGE_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif'}


def _ensure_image_extension(file_path: str) -> str:
    """Ensure file path has a valid image extension, defaulting to .png.

    Parameters
    ----------
    file_path : str
        The original file path.

    Returns
    -------
    str
        File path with a valid image extension.
    """
    ext = os.path.splitext(file_path)[1].lower()
    if ext in _VALID_IMAGE_EXTENSIONS:
        return file_path
    return f"{file_path}.png"
from PySide6.QtCore import Qt, QTimer, QThread, Signal, QSettings
from PySide6.QtGui import QAction, QKeySequence

from .qt_image_panel import ImagePanel
from .qt_param_panel import CameraParamPanel
from .theme import set_app_theme, get_current_theme, get_initial_theme, ThemeName
from .export_dialog import ExportDialog
from core.rectifier import StereoRectifier
from core.depth import DepthEstimator, normalize_stereo_pair


class StereoCalibrationGUI(QMainWindow):
    """Main Qt6 application window for stereo camera calibration and depth estimation."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Stereo Camera Calibration & Depth Toolbox")
        self.setGeometry(100, 100, 1600, 900)
        self.setMinimumSize(1200, 700)

        self.rectifier = StereoRectifier()
        self.depth_estimator = DepthEstimator()

        self.left_image_path: Optional[str] = None
        self.right_image_path: Optional[str] = None

        self._depth_debounce_timer = QTimer(self)
        self._depth_debounce_timer.setSingleShot(True)
        self._depth_debounce_timer.timeout.connect(self._update_depth)

        self._update_debounce_timer = QTimer(self)
        self._update_debounce_timer.setSingleShot(True)
        self._update_debounce_timer.timeout.connect(self._update_rectification)

        try:
            from core.raft_stereo_check import check_raft_available
            self.raft_available, self.raft_unavailable_reason = check_raft_available()
        except Exception:
            self.raft_available = False
            self.raft_unavailable_reason = "Unknown error"

        self._shared_zoom_factor: float = 1.0
        self._shared_zoom_locked: bool = False  # prevents recursive zoom updates

        self._settings = QSettings("StereoDepthToolbox", "StereoCalibrationGUI")
        self._create_menu_bar()
        self._create_ui()
        self._bind_shortcuts()
        self._apply_saved_theme()

    def _bind_shortcuts(self):
        """Bind keyboard shortcuts."""
        save_action = QAction(self)
        save_action.setShortcut(QKeySequence("Ctrl+S"))
        save_action.triggered.connect(self._save_rectified_images)

        open_left_action = QAction(self)
        open_left_action.setShortcut(QKeySequence("Ctrl+O"))
        open_left_action.triggered.connect(self._load_left_image)

        open_right_action = QAction(self)
        open_right_action.setShortcut(QKeySequence("Ctrl+R"))
        open_right_action.triggered.connect(self._load_right_image)

        update_action = QAction(self)
        update_action.setShortcut(QKeySequence("F5"))
        update_action.triggered.connect(self._update_rectification)

    def _create_menu_bar(self):
        """Create the application menu bar."""
        menubar = self.menuBar()

        # File menu
        file_menu = menubar.addMenu("&File")

        # Load images
        load_left_action = QAction("Load Left Image...", self)
        load_left_action.setShortcut("Ctrl+O")
        load_left_action.triggered.connect(self._load_left_image)
        file_menu.addAction(load_left_action)

        load_right_action = QAction("Load Right Image...", self)
        load_right_action.setShortcut("Ctrl+R")
        load_right_action.triggered.connect(self._load_right_image)
        file_menu.addAction(load_right_action)

        file_menu.addSeparator()

        # Calibration
        save_cal_action = QAction("Save Calibration...", self)
        save_cal_action.setShortcut("Ctrl+Shift+S")
        save_cal_action.triggered.connect(self._save_calibration)
        file_menu.addAction(save_cal_action)

        load_cal_action = QAction("Load Calibration...", self)
        load_cal_action.setShortcut("Ctrl+L")
        load_cal_action.triggered.connect(self._load_calibration)
        file_menu.addAction(load_cal_action)

        file_menu.addSeparator()

        # Export
        export_rectified_action = QAction("Export Rectified Images...", self)
        export_rectified_action.triggered.connect(self._save_rectified_images)
        file_menu.addAction(export_rectified_action)

        export_depth_action = QAction("Export Depth Map...", self)
        export_depth_action.triggered.connect(self._save_depth_map)
        file_menu.addAction(export_depth_action)

        file_menu.addSeparator()

        # Exit
        quit_action = QAction("E&xit", self)
        quit_action.setShortcut("Ctrl+Q")
        quit_action.triggered.connect(self.close)
        file_menu.addAction(quit_action)

        # View menu
        view_menu = menubar.addMenu("&View")

        # Theme toggle action
        self._theme_action = QAction("&Dark Theme", self, checkable=True, checked=True)
        self._theme_action.setShortcut("Ctrl+T")
        self._theme_action.triggered.connect(self._toggle_theme)
        view_menu.addAction(self._theme_action)

        # Refresh action
        refresh_action = QAction("&Refresh", self)
        refresh_action.setShortcut("F5")
        refresh_action.triggered.connect(self._update_rectification)
        view_menu.addAction(refresh_action)

        # Help menu
        help_menu = menubar.addMenu("&Help")
        about_action = QAction("&About", self)
        about_action.triggered.connect(self._show_about)
        help_menu.addAction(about_action)

    def _apply_saved_theme(self):
        """Load and apply the saved theme preference or system default."""
        app = QApplication.instance()
        initial_theme = get_initial_theme(self._settings, app)
        self._apply_theme(initial_theme)

    def _apply_theme(self, theme_name: ThemeName):
        """Apply a theme and persist the preference.

        Parameters
        ----------
        theme_name : "dark" | "light"
            Theme to apply.
        """
        # Apply to the entire application via the global app instance
        app = QApplication.instance()
        if app is not None:
            set_app_theme(theme_name, app)

        # Update menu bar check state
        if theme_name == "dark":
            self._theme_action.setChecked(True)
            self._theme_action.setText("&Dark Theme")
        else:
            self._theme_action.setChecked(False)
            self._theme_action.setText("&Light Theme")

        # Persist preference
        self._settings.setValue("ui/theme", theme_name)

    def _toggle_theme(self):
        """Toggle between dark and light themes."""
        current = get_current_theme()
        new_theme = "light" if current == "dark" else "dark"
        self._apply_theme(new_theme)

    def _show_about(self):
        """Show about dialog with software information."""
        about_text = (
            "<b>Stereo Rectification & Depth Toolbox</b><br><br>"
            "Author: Lars Kistner<br>"
            "License: BSD 3-Clause License<br><br>"
            "Source: <a href='https://github.com/Sr4l/stereo_rectify_and_depth_toolbox'>"
            "https://github.com/Sr4l/stereo_rectify_and_depth_toolbox</a><br><br>"
            "This toolbox provides stereo rectification, depth estimation using "
            "StereoBM, StereoSGBM, and RAFT-Stereo algorithms.<br><br>"
            "<i>Note: Depth calculation functionality is experimental and unverified.</i>"
        )
        QMessageBox.about(self, "About", about_text)

    def _create_ui(self):
        """Create the user interface."""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(5, 5, 5, 5)
        main_layout.setSpacing(5)

        # Main splitter with 3 columns
        main_splitter = QSplitter(Qt.Orientation.Horizontal)

        # Left panel - parameters
        left_panel = self._create_left_panel()
        main_splitter.addWidget(left_panel)

        # Center panel - rectified images
        center_panel = self._create_center_panel()
        main_splitter.addWidget(center_panel)

        # Right panel - depth and controls
        right_panel = self._create_right_panel()
        main_splitter.addWidget(right_panel)

        # Set initial splitter ratios (1:3:3)
        main_splitter.setSizes([300, 600, 600])

        main_layout.addWidget(main_splitter, 1)

        # Status bar
        self._create_status_bar()

    def _create_left_panel(self) -> QWidget:
        """Create left sidebar with parameter controls."""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)

        # Calibration buttons
        cal_group = QGroupBox("Calibration")
        cal_layout = QVBoxLayout(cal_group)
        cal_layout.setSpacing(5)

        self._btn_save_cal = QPushButton("Save Calibration")
        self._btn_save_cal.clicked.connect(self._save_calibration)
        cal_layout.addWidget(self._btn_save_cal)

        self._btn_load_cal = QPushButton("Load Calibration")
        self._btn_load_cal.clicked.connect(self._load_calibration)
        cal_layout.addWidget(self._btn_load_cal)

        layout.addWidget(cal_group)

        # Scrollable parameter area
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        scroll_content = QWidget()
        scroll_content.setObjectName("scrollContent")
        scroll_layout = QVBoxLayout(scroll_content)
        scroll_layout.setSpacing(5)

        self._left_param_panel = CameraParamPanel(
            scroll_content,
            title="Left Camera Parameters",
            on_change=self._on_param_change
        )
        scroll_layout.addWidget(self._left_param_panel)

        self._right_param_panel = CameraParamPanel(
            scroll_content,
            title="Right Camera Parameters",
            on_change=self._on_param_change
        )
        scroll_layout.addWidget(self._right_param_panel)

        scroll.setWidget(scroll_content)
        layout.addWidget(scroll, 1)

        return panel

    def _create_center_panel(self) -> QWidget:
        """Create center panel with all 4 zoom-synchronized image panels."""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(5)

        # === Row 1: Left/Right Camera Image Panels (50% of space) ===
        camera_group = QGroupBox("Camera Images")
        camera_layout = QVBoxLayout(camera_group)
        camera_layout.setSpacing(5)

        # Load buttons row
        load_btn_row = QHBoxLayout()
        load_btn_row.setAlignment(Qt.AlignmentFlag.AlignCenter)
        load_btn_row.setContentsMargins(10, 5, 10, 5)

        self._left_load_button = QPushButton("Load Left Image")
        self._left_load_button.clicked.connect(self._load_left_image)
        load_btn_row.addWidget(self._left_load_button)

        self._right_load_button = QPushButton("Load Right Image")
        self._right_load_button.clicked.connect(self._load_right_image)
        load_btn_row.addWidget(self._right_load_button)

        camera_layout.addLayout(load_btn_row)

        # Image panels row
        images_row = QHBoxLayout()
        images_row.setSpacing(5)

        self._left_camera_panel = ImagePanel(
            camera_group, "Left Camera", show_controls=False
        )
        images_row.addWidget(self._left_camera_panel, 1)

        self._right_camera_panel = ImagePanel(
            camera_group, "Right Camera", show_controls=False
        )
        images_row.addWidget(self._right_camera_panel, 1)

        camera_layout.addLayout(images_row)

        # Camera group takes 1 part of the space
        layout.addWidget(camera_group, 1)

        # === Row 2: Shared Zoom Control (minimal height) ===
        self._zoom_control_group = self._create_zoom_control_group()
        layout.addWidget(self._zoom_control_group)

        # === Row 3: Left/Right Rectified Image Panels (50% of space) ===
        rect_group = QGroupBox("Rectified Views")
        rect_layout = QVBoxLayout(rect_group)
        rect_layout.setSpacing(2)

        self._rectified_left_panel = ImagePanel(
            rect_group, "Left Rectified", show_controls=False
        )
        self._rectified_right_panel = ImagePanel(
            rect_group, "Right Rectified", show_controls=False
        )

        images_row = QHBoxLayout()
        images_row.addWidget(self._rectified_left_panel, 1)
        images_row.addWidget(self._rectified_right_panel, 1)
        rect_layout.addLayout(images_row)

        # === Row 4: Options ===
        options_row = QHBoxLayout()

        self._epipolar_checkbox = QCheckBox("Show Epipolar Lines")
        self._epipolar_checkbox.stateChanged.connect(self._toggle_epipolar_lines)
        options_row.addWidget(self._epipolar_checkbox)

        options_row.addWidget(QLabel("View:"))
        self._view_type_combo = QComboBox()
        self._view_type_combo.addItems(["rectified", "rectified gray"])
        self._view_type_combo.currentTextChanged.connect(self._update_rectification)
        options_row.addWidget(self._view_type_combo)

        options_row.addStretch(1)

        self._btn_save_rectified = QPushButton("Save Rectified Images")
        self._btn_save_rectified.clicked.connect(self._save_rectified_images)
        options_row.addWidget(self._btn_save_rectified)

        rect_layout.addLayout(options_row)
        # Rectified group takes 1 part of the space (equal to camera group = 50/50)
        layout.addWidget(rect_group, 1)

        return panel

    def _create_zoom_control_group(self) -> QGroupBox:
        """Create a shared zoom control group box with slider and buttons.
        
        Returns a QGroupBox containing a zoom slider that synchronizes zoom
        across all four image panels (camera left/right, rectified left/right).
        """
        zoom_group = QGroupBox("Synchronized Zoom Control")
        zoom_layout = QVBoxLayout(zoom_group)
        zoom_layout.setSpacing(3)

        # Slider row
        slider_row = QHBoxLayout()
        slider_row.addWidget(QLabel("Zoom:"))

        self._shared_zoom_slider = QSlider(Qt.Orientation.Horizontal)
        self._shared_zoom_slider.setRange(10, 500)
        self._shared_zoom_slider.setValue(100)
        self._shared_zoom_slider.setFixedWidth(250)
        self._shared_zoom_slider.valueChanged.connect(self._on_shared_zoom_changed)
        slider_row.addWidget(self._shared_zoom_slider)

        self._shared_zoom_label = QLabel("100%")
        self._shared_zoom_label.setFixedWidth(50)
        self._shared_zoom_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        slider_row.addWidget(self._shared_zoom_label)

        self._zoom_fit_button = QPushButton("Fit")
        self._zoom_fit_button.clicked.connect(self._on_zoom_fit)
        slider_row.addWidget(self._zoom_fit_button)

        self._zoom_1to1_button = QPushButton("1:1")
        self._zoom_1to1_button.clicked.connect(self._on_zoom_1to1)
        slider_row.addWidget(self._zoom_1to1_button)

        slider_row.addStretch(1)
        zoom_layout.addLayout(slider_row)

        return zoom_group

    def _on_shared_zoom_changed(self, value: int):
        """Handle shared zoom slider change and propagate to all panels."""
        if self._shared_zoom_locked:
            return
        self._shared_zoom_locked = True
        self._shared_zoom_factor = value / 100.0
        self._shared_zoom_label.setText(f"{value}%")
        self._apply_zoom_to_all_panels()
        self._shared_zoom_locked = False

    def _on_zoom_fit(self):
        """Handle Fit button - fit all panels to their viewport."""
        if self._shared_zoom_locked:
            return
        self._shared_zoom_locked = True
        panels = [
            self._left_camera_panel,
            self._right_camera_panel,
            self._rectified_left_panel,
            self._rectified_right_panel,
        ]
        for panel in panels:
            if hasattr(panel, 'fit_to_window'):
                panel.fit_to_window()
        # Sync slider to the resulting zoom level
        if panels[0]._zoom_slider is not None:
            self._shared_zoom_slider.setValue(int(panels[0]._zoom_factor * 100))
        self._shared_zoom_locked = False

    def _on_zoom_1to1(self):
        """Handle 1:1 button - reset all panels to 1:1 zoom."""
        if self._shared_zoom_locked:
            return
        self._shared_zoom_locked = True
        panels = [
            self._left_camera_panel,
            self._right_camera_panel,
            self._rectified_left_panel,
            self._rectified_right_panel,
        ]
        for panel in panels:
            if hasattr(panel, 'reset_zoom'):
                panel.reset_zoom()
        self._shared_zoom_slider.setValue(100)
        self._shared_zoom_locked = False

    def _setup_custom_slider(self, slider, from_val: int, to_val: int):
        """Configure slider for arrow key (+1) and mouse click (+5% of range) behavior.
        
        Args:
            slider: QSlider instance to configure
            from_val: Minimum value of the slider range
            to_val: Maximum value of the slider range
        """
        slider.setSingleStep(1)  # Arrow keys: +/-1
        range_size = to_val - from_val
        page_step = max(1, int(0.05 * range_size))  # Mouse click: +/-5% of range (rounded up to at least 1)
        slider.setPageStep(page_step)

    def _apply_zoom_to_all_panels(self):
        """Apply the shared zoom factor to all four image panels."""
        zoom_factor = self._shared_zoom_factor
        panels = [
            self._left_camera_panel,
            self._right_camera_panel,
            self._rectified_left_panel,
            self._rectified_right_panel,
        ]
        for panel in panels:
            if not hasattr(panel, '_zoom_factor') or panel._image is None:
                continue
            panel._zoom_factor = zoom_factor
            if hasattr(panel, '_zoom_slider') and panel._zoom_slider is not None:
                panel._zoom_slider.blockSignals(True)
                panel._zoom_slider.setValue(int(zoom_factor * 100))
                panel._zoom_slider.blockSignals(False)
            panel._update_display()

    def _create_right_panel(self) -> QWidget:
        """Create right panel with depth map and controls."""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(5)

        # Depth map panel
        depth_group = QGroupBox("Depth Map / Disparity")
        depth_layout = QVBoxLayout(depth_group)
        depth_layout.setContentsMargins(2, 2, 2, 2)

        self._depth_panel = ImagePanel(
            depth_group,
            "Depth Visualization",
            show_controls=True,
            value_callback=self._get_depth_value
        )
        self._depth_panel.set_export_callback(self._export_depth_data)
        depth_layout.addWidget(self._depth_panel)
        layout.addWidget(depth_group, 1)

        # Algorithm selector
        algo_group = QGroupBox("Stereo Matching Algorithm")
        algo_layout = QHBoxLayout(algo_group)
        algo_layout.setContentsMargins(5, 5, 5, 5)

        algo_layout.addWidget(QLabel("Algorithm:"))
        self._algorithm_combo = QComboBox()
        algo_values = ["BM", "SGBM"]
        if self.raft_available:
            algo_values.append("RAFT")
        self._algorithm_combo.addItems(algo_values)
        self._algorithm_combo.currentTextChanged.connect(self._on_algorithm_change)
        algo_layout.addWidget(self._algorithm_combo)

        algo_tooltip = "  BM: Fast | SGBM: Better"
        if self.raft_available:
            algo_tooltip += " | RAFT: DL SOTA"
        else:
            algo_tooltip += f" | RAFT: {self.raft_unavailable_reason}"
        algo_layout.addWidget(QLabel(algo_tooltip))
        layout.addWidget(algo_group)

        # BM controls
        self._bm_frame = ParamControlsGroup("StereoBM Parameters")
        bm_controls = [
            ('numDisparities', 'Num Disp:', 16, 256, 16, 16),
            ('blockSize', 'Block:', 5, 255, 2, 9),
            ('minDisparity', 'Min Disp:', -100, 100, 1, 0),
            ('uniquenessRatio', 'Unique:', 1, 100, 1, 10),
            ('speckleWindowSize', 'Speckle W:', 0, 200, 1, 100),
            ('speckleRange', 'Speckle R:', 0, 50, 1, 1),
        ]
        self._bm_sliders = {}
        for key, label, from_val, to_val, step, default in bm_controls:
            slider = QSlider(Qt.Orientation.Horizontal)
            slider.setRange(from_val, to_val)
            slider.setValue(default)
            self._setup_custom_slider(slider, from_val, to_val)
            self._bm_sliders[key] = (slider, default)
            self._bm_frame.add_control(label, slider, default)
            slider.valueChanged.connect(lambda: self._on_bm_param_change(key))
        self._bm_frame.layout().addStretch()
        self._bm_frame.add_hint("← → arrows: ±1  |  Click on track: ±5% of range")
        layout.addWidget(self._bm_frame.group)

        # SGBM controls
        self._sgbm_frame = ParamControlsGroup("StereoSGBM Parameters")
        sgbm_controls = [
            ('numDisparities', 'Num Disp:', 16, 256, 16, 16),
            ('blockSize', 'Block:', 5, 255, 2, 9),
            ('minDisparity', 'Min Disp:', -100, 100, 1, 0),
            ('uniquenessRatio', 'Unique:', 1, 100, 1, 10),
            ('P1', 'P1:', 0, 1000, 10, 200),
            ('P2', 'P2:', 0, 2000, 20, 400),
            ('preFilterCap', 'PreFilter:', 0, 100, 1, 31),
            ('speckleWindowSize', 'Speckle W:', 0, 200, 1, 100),
            ('speckleRange', 'Speckle R:', 0, 50, 1, 1),
        ]
        self._sgbm_sliders = {}
        for key, label, from_val, to_val, step, default in sgbm_controls:
            slider = QSlider(Qt.Orientation.Horizontal)
            slider.setRange(from_val, to_val)
            slider.setValue(default)
            self._setup_custom_slider(slider, from_val, to_val)
            self._sgbm_sliders[key] = (slider, default)
            self._sgbm_frame.add_control(label, slider, default)
            slider.valueChanged.connect(lambda: self._on_sgbm_param_change(key))
        self._sgbm_frame.layout().addStretch()
        self._sgbm_frame.add_hint("← → arrows: ±1  |  Click on track: ±5% of range")
        layout.addWidget(self._sgbm_frame.group)

        # RAFT controls
        self._raft_frame = ParamControlsGroup("RAFT-Stereo Parameters")
        raft_controls = [
            ('valid_iters', 'Iters:', 1, 64, 1, 32),
            ('n_downsample', 'Downsample:', 1, 3, 1, 2),
        ]
        self._raft_sliders = {}
        for key, label, from_val, to_val, step, default in raft_controls:
            slider = QSlider(Qt.Orientation.Horizontal)
            slider.setRange(from_val, to_val)
            slider.setValue(default)
            self._setup_custom_slider(slider, from_val, to_val)
            self._raft_sliders[key] = (slider, default)
            self._raft_frame.add_control(label, slider, default)
            slider.valueChanged.connect(lambda: self._on_raft_param_change(key))
        self._raft_frame.layout().addStretch()
        self._raft_frame.add_hint("← → arrows: ±1  |  Click on track: ±5% of range")

        # Model path row
        model_row = QHBoxLayout()
        model_row.addWidget(QLabel("Model Path:"))
        self._raft_model_path = "models/raftstereo-middlebury.pth"
        self._raft_model_entry = QLineEdit(self._raft_model_path)
        model_row.addWidget(self._raft_model_entry)
        browse_btn = QPushButton("Browse...")
        browse_btn.clicked.connect(self._browse_raft_model)
        model_row.addWidget(browse_btn)
        download_btn = QPushButton("Download...")
        download_btn.clicked.connect(self._download_raft_model)
        model_row.addWidget(download_btn)
        self._raft_frame.layout().addLayout(model_row)

        # Info label
        info_label = QLabel(
            "Note: GPU recommended. First load may take time."
        )
        info_label.setStyleSheet("color: yellow; font-size: 9pt;")
        info_label.setWordWrap(True)
        self._raft_frame.layout().addWidget(info_label)
        layout.addWidget(self._raft_frame.group)

        # Visualization controls (all in one row)
        vis_group = QGroupBox("Visualization Controls")
        vis_layout = QHBoxLayout(vis_group)
        vis_layout.setSpacing(10)

        self._btn_update_depth = QPushButton("Update Depth")
        self._btn_update_depth.clicked.connect(self._update_depth)
        vis_layout.addWidget(self._btn_update_depth)

        vis_layout.addWidget(QLabel("View:"))
        self._view_mode_combo = QComboBox()
        self._view_mode_combo.addItems(["disparity", "depth (m)"])
        self._view_mode_combo.currentTextChanged.connect(self._update_depth)
        vis_layout.addWidget(self._view_mode_combo)

        vis_layout.addWidget(QLabel("Colormap:"))
        self._colormap_combo = QComboBox()
        self._colormap_combo.addItems(["JET", "VIRIDIS", "MAGMA", "INFERNO", "PLASMA", "CIVIDIS"])
        self._colormap_combo.currentTextChanged.connect(self._update_depth)
        vis_layout.addWidget(self._colormap_combo)

        vis_layout.addStretch()

        layout.addWidget(vis_group)

        # Initially hide all algorithm panels, show only the default (BM)
        self._update_algorithm_visibility()

        return panel

    def _create_status_bar(self):
        """Create status bar at bottom."""
        self._status_label = QLabel("Ready")
        self.statusBar().addWidget(self._status_label, 1)
        self._time_label = QLabel("")
        self.statusBar().addPermanentWidget(self._time_label)

    def _on_param_change(self):
        """Handle parameter change with debouncing."""
        self._update_debounce_timer.stop()
        self._update_debounce_timer.start(500)

    def _update_rectification(self):
        """Update rectified images."""
        if self.rectifier.left_image is None or self.rectifier.right_image is None:
            return

        try:
            start_time = time.time()

            self.rectifier.set_left_intrinsics(
                self._left_param_panel.get_K(),
                self._left_param_panel.get_distortion()
            )
            self.rectifier.set_right_intrinsics(
                self._right_param_panel.get_K(),
                self._right_param_panel.get_distortion()
            )
            self.rectifier.set_extrinsics(
                self._right_param_panel.get_R(),
                self._right_param_panel.get_T()
            )

            rect_left, rect_right = self.rectifier.rectify()

            view_type = self._view_type_combo.currentText()

            if view_type == "rectified gray":
                if rect_left is not None and rect_right is not None:
                    left_gray, right_gray = normalize_stereo_pair(rect_left, rect_right)
                    display_left = cv2.cvtColor(left_gray, cv2.COLOR_GRAY2BGR)
                    display_right = cv2.cvtColor(right_gray, cv2.COLOR_GRAY2BGR)
                else:
                    display_left = rect_left if rect_left is not None else self.rectifier.left_image
                    display_right = rect_right if rect_right is not None else self.rectifier.right_image
            else:
                display_left = rect_left if rect_left is not None else self.rectifier.left_image
                display_right = rect_right if rect_right is not None else self.rectifier.right_image

            if display_left is not None and display_right is not None:
                if self._epipolar_checkbox.isChecked():
                    display_left = self.rectifier.draw_epipolar_lines(display_left)
                    display_right = self.rectifier.draw_epipolar_lines(display_right)

                # Update rectified panels
                self._rectified_left_panel.set_image(display_left)
                self._rectified_right_panel.set_image(display_right)

                # Re-apply shared zoom to all panels
                self._apply_zoom_to_all_panels()

                self._update_depth()

            elapsed = (time.time() - start_time) * 1000
            self._time_label.setText(f"Rectification: {elapsed:.1f}ms")
            self._status_label.setText("Rectification updated")

        except Exception as e:
            error_msg = str(e)
            if 'stereoRectify' in error_msg or 'nt > 0.0' in error_msg:
                error_msg = (
                    "Invalid camera parameters. Please check:\n"
                    "- Focal lengths (fx, fy) must be positive\n"
                    "- Principal point (cx, cy) should be within image bounds"
                )
            self._status_label.setText(f"Error: {error_msg}")

    def _toggle_epipolar_lines(self):
        """Toggle epipolar lines display."""
        self._update_rectification()

    def _load_left_image(self):
        """Load left camera image."""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Left Image",
            "",
            "Image files (*.jpg *.jpeg *.png *.bmp *.tiff *.tif);;All files (*)",
        )
        if file_path:
            self.left_image_path = file_path
            image = cv2.imread(file_path)
            if image is not None:
                self.rectifier.set_left_image(image)
                self._left_camera_panel.set_image(image)
                self._update_rectification()
                self._status_label.setText(f"Left image loaded: {os.path.basename(file_path)}")

    def _load_right_image(self):
        """Load right camera image."""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Right Image",
            "",
            "Image files (*.jpg *.jpeg *.png *.bmp *.tiff *.tif);;All files (*)",
        )
        if file_path:
            self.right_image_path = file_path
            image = cv2.imread(file_path)
            if image is not None:
                self.rectifier.set_right_image(image)
                self._right_camera_panel.set_image(image)
                self._update_rectification()
                self._status_label.setText(f"Right image loaded: {os.path.basename(file_path)}")

    def _browse_raft_model(self):
        """Browse for RAFT-Stereo model checkpoint."""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select RAFT-Stereo Model",
            "",
            "PyTorch models (*.pth);;All files (*)",
        )
        if file_path:
            self._raft_model_entry.setText(file_path)

    def _download_raft_model(self):
        """Download pretrained RAFT-Stereo model."""
        try:
            from core.depth import TORCH_AVAILABLE
            if not TORCH_AVAILABLE:
                QMessageBox.critical(
                    self,
                    "Error",
                    "PyTorch not installed. Please install PyTorch first:\n\npip install torch torchvision"
                )
                return
        except Exception:
            QMessageBox.critical(self, "Error", "Cannot check PyTorch availability.")
            return

        # Create dialog
        dialog = QDialog(self)
        dialog.setWindowTitle("Download RAFT-Stereo Model")
        dialog.resize(450, 250)
        # Transient relationship is set via parent constructor (PySide6 compatible)

        layout = QVBoxLayout(dialog)

        title_label = QLabel("Select model to download:")
        title_label.setStyleSheet("font-size: 11pt; font-weight: bold;")
        layout.addWidget(title_label)

        model_info = {
            'middlebury': 'Middlebury - Best for in-the-wild images (RECOMMENDED)',
            'eth3d': 'ETH3D - High resolution stereo',
            'sceneflow': 'SceneFlow - General purpose (FlyingThings3D)',
            'realtime': 'Realtime - Fastest model',
        }

        radio_buttons = {}
        selected = ['middlebury']

        for key, desc in model_info.items():
            rb = QRadioButton(desc)
            if key == 'middlebury':
                rb.setChecked(True)
            radio_buttons[key] = rb
            layout.addWidget(rb)

        btn_row = QHBoxLayout()

        def on_download():
            for key, rb in radio_buttons.items():
                if rb.isChecked():
                    selected[0] = key
                    break
            dialog.accept()

        download_btn = QPushButton("Download")
        download_btn.clicked.connect(on_download)
        btn_row.addWidget(download_btn)

        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(dialog.reject)
        btn_row.addWidget(cancel_btn)

        layout.addLayout(btn_row)
        dialog.setLayout(layout)

        result = dialog.exec()

        if result == 1:
            self._download_raft_model_script(selected[0])

    def _download_raft_model_script(self, model_name: str):
        """Run download script for RAFT-Stereo model."""
        models_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models')
        os.makedirs(models_dir, exist_ok=True)

        try:
            script_path = os.path.join(
                os.path.dirname(os.path.dirname(__file__)),
                'scripts',
                'download_raft_models.py'
            )

            result = subprocess.run(
                ['python', script_path, '--model', model_name, '--output', models_dir],
                capture_output=True,
                text=True,
                timeout=300
            )

            if result.returncode == 0:
                model_file = f"raftstereo-{model_name}.pth"
                self._raft_model_entry.setText(os.path.join(models_dir, model_file))
                QMessageBox.information(
                    self,
                    "Success",
                    f"Model downloaded successfully!\n\nSaved to: {os.path.join(models_dir, model_file)}"
                )
            else:
                QMessageBox.critical(self, "Error", f"Download failed:\n{result.stderr}")

        except subprocess.TimeoutExpired:
            QMessageBox.critical(self, "Error", "Download timed out. Please try again or download manually.")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Download failed: {str(e)}")

    def _on_algorithm_change(self):
        """Handle algorithm change."""
        algo = self._algorithm_combo.currentText()

        if algo == 'RAFT' and not self.raft_available:
            if "submodule" in self.raft_unavailable_reason.lower() or "not found" in self.raft_unavailable_reason.lower():
                self._show_raft_submodule_error()
            else:
                self._show_raft_error()
            self._algorithm_combo.setCurrentText("BM")
            algo = "BM"

        self.depth_estimator.set_algorithm(algo)
        self._update_algorithm_visibility()

        if algo == 'SGBM':
            params = {key: data[0].value() for key, data in self._sgbm_sliders.items()}
            self.depth_estimator.set_sgbm_params(**params)
        elif algo == 'RAFT':
            params = {key: data[0].value() for key, data in self._raft_sliders.items()}
            self.depth_estimator.set_raft_params(**params)
        else:
            params = {key: data[0].value() for key, data in self._bm_sliders.items()}
            self.depth_estimator.set_bm_params(**params)

        self._update_depth()

    def _update_algorithm_visibility(self):
        """Show/hide parameter panels based on selected algorithm."""
        algo = self._algorithm_combo.currentText()

        # Hide all first
        self._bm_frame.group.setVisible(False)
        self._sgbm_frame.group.setVisible(False)
        self._raft_frame.group.setVisible(False)

        if algo == 'BM':
            self._bm_frame.group.setVisible(True)
        elif algo == 'SGBM':
            self._sgbm_frame.group.setVisible(True)
        elif algo == 'RAFT':
            self._raft_frame.group.setVisible(True)

    def _on_bm_param_change(self, key: str):
        """Handle BM parameter change with debouncing."""
        if self._algorithm_combo.currentText() != 'BM':
            return
        self._depth_debounce_timer.start(300)

    def _on_sgbm_param_change(self, key: str):
        """Handle SGBM parameter change with debouncing."""
        if self._algorithm_combo.currentText() != 'SGBM':
            return
        self._depth_debounce_timer.start(300)

    def _on_raft_param_change(self, key: str):
        """Handle RAFT parameter change with debouncing."""
        if self._algorithm_combo.currentText() != 'RAFT':
            return
        self._depth_debounce_timer.start(300)

    def _update_depth(self):
        """Update depth map."""
        if self.rectifier.rectified_left is None or self.rectifier.rectified_right is None:
            return

        try:
            start_time = time.time()

            algo = self._algorithm_combo.currentText()

            if algo == 'SGBM':
                params = {key: data[0].value() for key, data in self._sgbm_sliders.items()}
                if 'blockSize' in params:
                    if params['blockSize'] % 2 == 0:
                        params['blockSize'] += 1
                    params['blockSize'] = max(5, min(255, params['blockSize']))
                self.depth_estimator.set_sgbm_params(**params)
                disparity = self.depth_estimator.compute_disparity_sgbm(
                    self.rectifier.rectified_left,
                    self.rectifier.rectified_right
                )
            elif algo == 'RAFT':
                params = {key: data[0].value() for key, data in self._raft_sliders.items()}
                model_path = self._raft_model_entry.text()
                self.depth_estimator.set_raft_params(**params)

                try:
                    from core.depth import TORCH_AVAILABLE
                    if not TORCH_AVAILABLE:
                        self._status_label.setText("Error: PyTorch not installed. Cannot use RAFT-Stereo.")
                        QMessageBox.critical(
                            self,
                            "Error",
                            "PyTorch not installed.\n\nPlease install PyTorch:\npip install torch torchvision"
                        )
                        return
                except Exception:
                    self._status_label.setText("Error: Cannot check PyTorch availability.")
                    return

                if not os.path.exists(model_path):
                    self._status_label.setText("Error: RAFT model not found")
                    QMessageBox.critical(
                        self,
                        "Model Not Found",
                        f"RAFT-Stereo model file not found:\n{model_path}\n\n"
                        f"Please download a model using:\n"
                        f"  • The 'Download...' button in the RAFT panel\n"
                        f"  • Or run: python scripts/download_raft_models.py --model middlebury"
                    )
                    return

                disparity = self.depth_estimator.compute_disparity_raft(
                    self.rectifier.rectified_left,
                    self.rectifier.rectified_right,
                    model_path=model_path
                )
            else:
                params = {key: data[0].value() for key, data in self._bm_sliders.items()}
                if 'blockSize' in params:
                    if params['blockSize'] % 2 == 0:
                        params['blockSize'] += 1
                    params['blockSize'] = max(5, min(255, params['blockSize']))
                self.depth_estimator.set_bm_params(**params)
                disparity = self.depth_estimator.compute_disparity(
                    self.rectifier.rectified_left,
                    self.rectifier.rectified_right
                )

            if disparity is not None:
                colormap_name = self._colormap_combo.currentText()
                colormap_map = {
                    'JET': cv2.COLORMAP_JET,
                    'VIRIDIS': cv2.COLORMAP_VIRIDIS,
                    'MAGMA': cv2.COLORMAP_MAGMA,
                    'INFERNO': cv2.COLORMAP_INFERNO,
                    'PLASMA': cv2.COLORMAP_PLASMA,
                    'CIVIDIS': cv2.COLORMAP_CIVIDIS,
                }
                colormap = colormap_map.get(colormap_name, cv2.COLORMAP_JET)

                view_mode = self._view_mode_combo.currentText()

                if view_mode == "depth (m)":
                    # Set camera params from the rectification translation vector
                    # Baseline = magnitude of translation vector (after rectification)
                    # IMPORTANT: T must be in meters for depth formula to work correctly
                    # If your calibration used different units (e.g., mm), convert here.
                    T_right = self._right_param_panel.get_T()
                    baseline_m = float(np.linalg.norm(T_right))
                    K_left = self._left_param_panel.get_K()
                    # Use fx (horizontal focal length) since disparity is computed along x-axis
                    # This is more accurate than averaging fx and fy for cameras with
                    # non-square pixels where fx != fy
                    focal_length_px = K_left[0, 0]  # fx
                    self.depth_estimator.set_camera_params(baseline_m, focal_length_px)

                    # Always compute depth_map so the tooltip can read from it
                    # Depth formula: depth (meters) = (baseline_meters * focal_length_pixels) / disparity_pixels
                    depth_map = self.depth_estimator.compute_depth()
                    if depth_map is not None:
                        # Use meters consistently - clip to 100 meters maximum for visualization
                        depth_m_clipped = np.clip(depth_map, 0, 100.0)
                        depth_normalized = (
                            ((depth_m_clipped - depth_m_clipped.min()) /
                             (depth_m_clipped.max() - depth_m_clipped.min()) * 255)
                            .astype(np.uint8)
                        )
                        colored_depth = cv2.applyColorMap(depth_normalized, colormap)
                        self._depth_panel.set_image(colored_depth)

                    stats = self.depth_estimator.get_depth_stats()
                    self._status_label.setText(
                        f"[{algo}] Depth - Min: {stats['min']:.3f}m, "
                        f"Max: {stats['max']:.3f}m, Mean: {stats['mean']:.3f}m"
                    )
                else:
                    colored_disparity = self.depth_estimator.apply_colormap(disparity, colormap)
                    self._depth_panel.set_image(colored_disparity)

                    stats = self.depth_estimator.get_disparity_stats()
                    self._status_label.setText(
                        f"[{algo}] Disparity - Min: {stats['min']:.2f}, "
                        f"Max: {stats['max']:.2f}, Mean: {stats['mean']:.2f}"
                    )

            elapsed = (time.time() - start_time) * 1000
            self._time_label.setText(f"Depth: {elapsed:.1f}ms")

        except Exception as e:
            self._status_label.setText(f"Depth Error: {str(e)}")

    def _get_depth_value(self, x: int, y: int) -> str:
        """Get depth/disparity value at pixel coordinates for tooltip.

        Reads from the pre-computed depth_estimator.depth_map when in depth mode,
        avoiding redundant calculation of (baseline * focal_length) / disparity.
        """
        if (self.depth_estimator.disparity is None or
                y >= self.depth_estimator.disparity.shape[0] or
                x >= self.depth_estimator.disparity.shape[1]):
            return None

        try:
            disp_val = self.depth_estimator.disparity[y, x]

            if disp_val <= 0:
                return f"({x}, {y})  Disparity: 0.00  Depth: N/A"

            view_mode = self._view_mode_combo.currentText()

            if view_mode == "depth (m)":
                # Ensure camera params are set (should already be set by _update_depth)
                # Use same calculation as _update_depth for consistency
                T_right = self._right_param_panel.get_T()
                baseline_m = float(np.linalg.norm(T_right))
                K_left = self._left_param_panel.get_K()
                focal_length_px = K_left[0, 0]  # fx - horizontal focal length

                # Only update if params changed (avoids redundant set calls)
                if (abs(self.depth_estimator.baseline - baseline_m) > 1e-9 or
                        abs(self.depth_estimator.focal_length - focal_length_px) > 1e-9):
                    self.depth_estimator.set_camera_params(baseline_m, focal_length_px)
                    # Re-compute depth_map with updated params
                    self.depth_estimator.compute_depth()

                # Read depth directly from the pre-computed depth_map
                if (self.depth_estimator.depth_map is not None and
                        y < self.depth_estimator.depth_map.shape[0] and
                        x < self.depth_estimator.depth_map.shape[1]):
                    depth_m = self.depth_estimator.depth_map[y, x]
                else:
                    return None

                # Clip display to 100 meters maximum
                if depth_m > 100.0:
                    return f"({x}, {y})  Disparity: {disp_val:.2f}  Depth: >100m"

                return f"({x}, {y})  Disparity: {disp_val:.2f}  Depth: {depth_m:.3f}m"
            else:
                return f"({x}, {y})  Disparity: {disp_val:.2f}"

        except Exception:
            return None

    def _save_rectified_images(self):
        """Save rectified images to files."""
        if self.rectifier.rectified_left is None:
            QMessageBox.warning(self, "Warning", "No rectified images to save")
            return

        left_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Left Rectified Image",
            "",
            "PNG files (*.png);;JPEG files (*.jpg)",
        )
        if left_path:
            left_path = _ensure_image_extension(left_path)
            cv2.imwrite(left_path, self.rectifier.rectified_left)

        right_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Right Rectified Image",
            "",
            "PNG files (*.png);;JPEG files (*.jpg)",
        )
        if right_path:
            right_path = _ensure_image_extension(right_path)
            cv2.imwrite(right_path, self.rectifier.rectified_right)
            self._status_label.setText("Rectified images saved")

    def _save_depth_map(self):
        """Save depth map to file."""
        if self.depth_estimator.disparity is None:
            QMessageBox.warning(self, "Warning", "No depth map to save")
            return

        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Depth Map",
            "",
            "PNG files (*.png);;JPEG files (*.jpg)",
        )

        if file_path:
            file_path = _ensure_image_extension(file_path)
            colormap_name = self._colormap_combo.currentText()
            colormap_map = {
                'JET': cv2.COLORMAP_JET,
                'VIRIDIS': cv2.COLORMAP_VIRIDIS,
                'MAGMA': cv2.COLORMAP_MAGMA,
                'INFERNO': cv2.COLORMAP_INFERNO,
                'PLASMA': cv2.COLORMAP_PLASMA,
                'CIVIDIS': cv2.COLORMAP_CIVIDIS,
            }
            colormap = colormap_map.get(colormap_name, cv2.COLORMAP_JET)

            colored = self.depth_estimator.apply_colormap(self.depth_estimator.disparity, colormap)
            cv2.imwrite(file_path, colored)
            self._status_label.setText("Depth map saved")

    def _save_calibration(self):
        """Save calibration parameters to JSON file."""
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Calibration",
            "",
            "JSON files (*.json)",
        )

        if file_path:
            calibration = {
                'left_camera': {
                    'K': self._left_param_panel.get_K().tolist(),
                    'distortion': self._left_param_panel.get_distortion().tolist(),
                },
                'right_camera': {
                    'K': self._right_param_panel.get_K().tolist(),
                    'distortion': self._right_param_panel.get_distortion().tolist(),
                    'R': self._right_param_panel.get_R().tolist(),
                    'T': self._right_param_panel.get_T().tolist(),
                },
            }

            with open(file_path, 'w') as f:
                json.dump(calibration, f, indent=2)

            self._status_label.setText(f"Calibration saved to {os.path.basename(file_path)}")

    def _load_calibration(self):
        """Load calibration parameters from JSON file."""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Load Calibration",
            "",
            "JSON files (*.json)",
        )

        if file_path:
            try:
                with open(file_path, 'r') as f:
                    calibration = json.load(f)

                left_cam = calibration['left_camera']
                right_cam = calibration['right_camera']

                self._left_param_panel.set_K(np.array(left_cam['K']))
                self._left_param_panel.set_distortion(np.array(left_cam['distortion']))

                self._right_param_panel.set_K(np.array(right_cam['K']))
                self._right_param_panel.set_distortion(np.array(right_cam['distortion']))
                self._right_param_panel.set_R(np.array(right_cam['R']))
                self._right_param_panel.set_T(np.array(right_cam['T']))

                self._update_rectification()
                self._status_label.setText(
                    f"Calibration loaded from {os.path.basename(file_path)}"
                )

            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to load calibration: {str(e)}")

    def _show_raft_error(self):
        """Show error dialog for missing PyTorch."""
        message = (
            "RAFT-Stereo requires PyTorch which is not installed.\n\n"
            "Please install PyTorch:\n"
            "  pip install torch torchvision\n\n"
            "Switching to BM algorithm."
        )
        QMessageBox.critical(self, "PyTorch Not Installed", message)

    def _show_raft_submodule_error(self):
        """Show error dialog for uninitialized RAFT-Stereo submodule."""
        message = (
            f"RAFT-Stereo submodule is not initialized.\n\n"
            f"Reason: {self.raft_unavailable_reason}\n\n"
            f"To fix this, run the following command in your terminal:\n\n"
            f"  git submodule update --init --recursive\n\n"
            f"This will download the RAFT-Stereo code from GitHub.\n\n"
            f"Switching to BM algorithm."
        )
        QMessageBox.critical(self, "RAFT-Stereo Submodule Not Initialized", message)

    def _export_depth_data(self):
        """Export depth or disparity data to file."""
        if self.depth_estimator.disparity is None:
            QMessageBox.warning(self, "Warning", "No depth/disparity data to export. Compute depth first.")
            return

        disparity = self.depth_estimator.disparity
        height, width = disparity.shape

        # Show export dialog
        dialog = ExportDialog(self, data_shape=(height, width))
        if dialog.exec() != 1:  # Accepted
            return

        file_path = dialog.get_file_path()
        export_type = dialog.get_export_type()
        file_format = dialog.get_format()

        if not file_path:
            return

        # Ensure correct extension
        ext_map = {
            "npy": ".npy",
            "mat": ".mat",
            "tiff": ".tiff",
            "csv": ".csv",
        }
        ext = ext_map.get(file_format, "")
        if not file_path.lower().endswith(ext.lower()):
            file_path += ext

        try:
            if export_type == "depth":
                depth_map = self.depth_estimator.compute_depth()
                if depth_map is None:
                    QMessageBox.warning(self, "Warning", "Cannot compute depth data. Check camera parameters.")
                    return
                data_to_export = depth_map
            else:
                data_to_export = disparity

            if file_format == "npy":
                np.save(file_path, data_to_export)
            elif file_format == "mat":
                self._save_mat(file_path, data_to_export, export_type)
            elif file_format == "tiff":
                self._save_tiff(file_path, data_to_export)
            elif file_format == "csv":
                self._save_csv(file_path, data_to_export)

            self._status_label.setText(f"Data exported to {os.path.basename(file_path)}")

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to export data: {str(e)}")

    def _save_mat(self, file_path: str, data: np.ndarray, export_type: str):
        """Save data as MATLAB .mat file using scipy."""
        try:
            from scipy import io
        except ImportError:
            QMessageBox.critical(
                self,
                "Error",
                "scipy is required for MATLAB .mat format.\nInstall with: pip install scipy"
            )
            return

        variable_name = "depth" if export_type == "depth" else "disparity"
        io.savemat(file_path, {variable_name: data})

    def _save_tiff(self, file_path: str, data: np.ndarray):
        """Save data as TIFF file."""
        try:
            from PIL import Image
        except ImportError:
            QMessageBox.critical(
                self,
                "Error",
                "Pillow is required for TIFF export.\nInstall with: pip install Pillow"
            )
            return

        # Normalize data to 0-65535 for 16-bit TIFF
        data_min = data.min()
        data_max = data.max()
        data_range = data_max - data_min

        if data_range == 0:
            normalized = np.zeros(data.shape, dtype=np.uint16)
        else:
            normalized = ((data - data_min) / data_range * 65535).astype(np.uint16)

        img = Image.fromarray(normalized, mode='I;16')
        img.save(file_path)

    def _save_csv(self, file_path: str, data: np.ndarray):
        """Save data as CSV with one column per pixel column."""
        # Handle NaN/inf values by converting to string-friendly format
        clean_data = np.where(
            np.isfinite(data), data, np.nan
        )
        np.savetxt(file_path, clean_data, delimiter=',', fmt='%.6f')


class ParamControlsGroup:
    """Helper class to create a group box with parameter sliders."""

    def __init__(self, title: str):
        self.group = QGroupBox(title)
        self._layout = QVBoxLayout(self.group)
        self._layout.setContentsMargins(5, 5, 5, 5)
        self._layout.setSpacing(2)

    def add_control(self, label_text: str, slider: QSlider, default_value: int):
        """Add a labeled slider to the controls."""
        row = QWidget()
        row_layout = QHBoxLayout(row)
        row_layout.setContentsMargins(0, 0, 0, 0)
        row_layout.setSpacing(5)

        lbl = QLabel(label_text)
        lbl.setFixedWidth(80)
        lbl.setStyleSheet("font-weight: bold;")
        row_layout.addWidget(lbl)

        # Use QLineEdit instead of QLabel so user can type values
        value_entry = QLineEdit(str(default_value))
        value_entry.setFixedWidth(55)
        value_entry.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        value_entry.setAttribute(Qt.WidgetAttribute.WA_InputMethodEnabled, True)

        # Slider -> entry sync
        slider.valueChanged.connect(lambda v, el=value_entry: el.setText(str(v)))
        # Entry -> slider sync (on Enter/Return)
        value_entry.returnPressed.connect(
            lambda el=value_entry, sl=slider: self._sync_entry_to_slider(el, sl)
        )

        row_layout.addWidget(slider, 1)
        row_layout.addWidget(value_entry)

        self._layout.addWidget(row)

    def _sync_entry_to_slider(self, entry: QLineEdit, slider: QSlider):
        """Sync the QLineEdit value to the QSlider, clamping to slider range."""
        try:
            val = int(entry.text())
            clamped = max(slider.minimum(), min(slider.maximum(), val))
            slider.setValue(clamped)
            entry.setText(str(clamped))
        except ValueError:
            # Invalid input: revert to current slider value
            entry.setText(str(slider.value()))

    def layout(self) -> QVBoxLayout:
        """Return the internal layout."""
        return self._layout

    def add_hint(self, hint_text: str):
        """Add a hint label at the bottom explaining slider behavior."""
        hint = QLabel(hint_text)
        hint.setStyleSheet("font-size: 8pt;")
        hint.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._layout.addWidget(hint)


def main():
    """Main entry point."""
    import sys
    app = QApplication.instance() or QApplication(sys.argv)
    app.setStyle("Fusion")  # Use Fusion style for consistent appearance across platforms
    app.setApplicationName("Stereo Camera Calibration & Depth Toolbox")

    window = StereoCalibrationGUI()
    window.show()

    sys.exit(app.exec())


if __name__ == '__main__':
    main()