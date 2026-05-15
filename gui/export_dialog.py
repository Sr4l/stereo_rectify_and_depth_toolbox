"""Export dialog for depth/disparity data."""

import os
from typing import Optional

from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QGroupBox,
    QLabel, QRadioButton, QLineEdit, QPushButton,
    QComboBox, QFileDialog, QMessageBox
)
from PySide6.QtCore import Qt


class ExportDialog(QDialog):
    """Dialog to configure export of depth/disparity data.

    Parameters
    ----------
    parent : QWidget
        Parent widget.
    data_shape : tuple
        Shape of the data array (height, width).
    """

    def __init__(self, parent=None, data_shape: tuple = (0, 0)):
        super().__init__(parent)
        self.setWindowTitle("Export Depth/Disparity Data")
        self.setModal(True)
        self.setMinimumWidth(450)
        self._height, self._width = data_shape
        self._selected_format = "npy"
        self._selected_type = "disparity"
        self._setup_ui()

    def _setup_ui(self):
        """Build the dialog UI."""
        layout = QVBoxLayout(self)
        layout.setSpacing(12)

        # Title
        title = QLabel("Export Depth/Disparity Data")
        title.setStyleSheet("font-size: 12pt; font-weight: bold;")
        layout.addWidget(title)

        # Info label
        info = QLabel(f"Data dimensions: {self._height} × {self._width} pixels")
        info.setStyleSheet("color: gray; font-size: 9pt;")
        layout.addWidget(info)

        # Export type group
        type_group = QGroupBox("Export Type")
        type_layout = QVBoxLayout(type_group)
        type_layout.setSpacing(4)

        self._disparity_radio = QRadioButton("Disparity values")
        self._disparity_radio.setChecked(True)
        type_layout.addWidget(self._disparity_radio)

        self._depth_radio = QRadioButton("Depth values (mm)")
        type_layout.addWidget(self._depth_radio)

        layout.addWidget(type_group)

        # File format group
        format_group = QGroupBox("File Format")
        format_layout = QHBoxLayout(format_group)
        format_layout.setSpacing(8)

        format_layout.addWidget(QLabel("Format:"))
        self._format_combo = QComboBox()
        self._format_combo.addItems(["npy", "mat", "tiff", "csv"])
        self._format_combo.currentTextChanged.connect(self._on_format_changed)
        format_layout.addWidget(self._format_combo)

        layout.addWidget(format_group)

        # Format description
        self._format_desc = QLabel(
            "NumPy binary format. Fast and preserves full precision."
        )
        self._format_desc.setWordWrap(True)
        self._format_desc.setStyleSheet("font-size: 9pt; color: gray;")
        layout.addWidget(self._format_desc)

        # File name group
        file_group = QGroupBox("File Name")
        file_layout = QHBoxLayout(file_group)
        file_layout.setSpacing(6)

        self._name_edit = QLineEdit()
        self._name_edit.setText(self._default_filename())
        file_layout.addWidget(self._name_edit)

        browse_btn = QPushButton("Browse...")
        browse_btn.clicked.connect(self._browse_filename)
        file_layout.addWidget(browse_btn)

        layout.addWidget(file_group)

        layout.addStretch()

        # Buttons
        btn_layout = QHBoxLayout()
        btn_layout.addStretch()

        self._export_btn = QPushButton("Export")
        self._export_btn.clicked.connect(self._on_export)
        btn_layout.addWidget(self._export_btn)

        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        btn_layout.addWidget(cancel_btn)

        layout.addLayout(btn_layout)

    def _default_filename(self) -> str:
        """Generate default filename based on format and type."""
        type_suffix = "depth" if self._depth_radio.isChecked() else "disparity"
        return f"exported_{type_suffix}"

    def _browse_filename(self):
        """Open file browser to set export path."""
        format_filter_map = {
            "npy": "NumPy files (*.npy)",
            "mat": "MATLAB files (*.mat)",
            "tiff": "TIFF files (*.tiff *.tif)",
            "csv": "CSV files (*.csv)",
        }
        selected_format = self._format_combo.currentText()
        file_filter = format_filter_map.get(selected_format, "All files (*)")

        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Export Data",
            self._name_edit.text(),
            file_filter,
        )
        if file_path:
            # Ensure correct extension
            ext_map = {
                "npy": ".npy",
                "mat": ".mat",
                "tiff": ".tiff",
                "csv": ".csv",
            }
            ext = ext_map.get(selected_format, "")
            if not file_path.lower().endswith(ext.lower()):
                file_path += ext
            self._name_edit.setText(file_path)

    def _on_format_changed(self, format_name: str):
        """Update description when format changes."""
        self._selected_format = format_name.lower()
        descriptions = {
            "npy": "NumPy binary format. Fast and preserves full precision.",
            "mat": "MATLAB .mat format (requires scipy). Preserves full precision.",
            "tiff": "TIFF image format. May lose precision depending on data type.",
            "csv": "Comma-separated values. One row per pixel row, "
                   f"{self._width} columns per row.",
        }
        self._format_desc.setText(descriptions.get(self._selected_format, ""))

    def _on_export(self):
        """Validate inputs and accept dialog."""
        file_path = self._name_edit.text().strip()
        if not file_path:
            QMessageBox.warning(self, "Warning", "Please specify a file name.")
            return
        self.accept()

    def get_export_type(self) -> str:
        """Get selected export type."""
        return "depth" if self._depth_radio.isChecked() else "disparity"

    def get_file_path(self) -> str:
        """Get selected file path."""
        return self._name_edit.text().strip()

    def get_format(self) -> str:
        """Get selected file format."""
        return self._selected_format