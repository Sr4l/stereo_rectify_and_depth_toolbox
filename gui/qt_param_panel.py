import numpy as np
from typing import Callable, Optional, Dict, List

from PySide6.QtWidgets import (
    QWidget, QGroupBox, QVBoxLayout, QHBoxLayout, QGridLayout,
    QLineEdit, QLabel
)
from PySide6.QtCore import Signal


class CameraParamPanel(QWidget):
    """Panel for editing camera intrinsic and extrinsic parameters using Qt widgets."""

    # Signal emitted when any parameter changes
    param_changed = Signal()

    def __init__(
        self,
        parent: QWidget = None,
        title: str = "Camera Parameters",
        on_change: Optional[Callable] = None,
    ):
        super().__init__(parent)
        self._on_change = on_change
        self._entries: Dict[str, List[List[QLineEdit]]] = {}
        self._dist_entries: Dict[str, QLineEdit] = {}
        self._t_entries: List[QLineEdit] = []

        # No per-widget stylesheet needed — global theme handles all styling

        group_box = QGroupBox(title)
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(5, 5, 5, 5)
        main_layout.addWidget(group_box)

        group_layout = QVBoxLayout(group_box)
        group_layout.setContentsMargins(10, 25, 10, 10)

        # Intrinsic matrix section
        k_frame = QWidget()
        k_layout = QVBoxLayout(k_frame)
        k_layout.setContentsMargins(0, 0, 0, 0)
        self._create_intrinsic_section(k_layout)
        group_layout.addWidget(k_frame)

        # Distortion section
        dist_frame = QWidget()
        dist_layout = QVBoxLayout(dist_frame)
        dist_layout.setContentsMargins(0, 0, 0, 0)
        self._create_distortion_section(dist_layout)
        group_layout.addWidget(dist_frame)

        # Extrinsic section
        ext_frame = QWidget()
        ext_layout = QVBoxLayout(ext_frame)
        ext_layout.setContentsMargins(0, 0, 0, 0)
        self._create_extrinsic_section(ext_layout)
        group_layout.addWidget(ext_frame)

    def _create_widgets(self):
        """Placeholder method - widgets are created inline in __init__."""
        pass

    def _create_intrinsic_section(self, parent_layout: QVBoxLayout):
        """Create intrinsic camera parameters (fx, fy, cx, cy) input section."""
        k_group = QGroupBox("Intrinsic Camera Parameters")
        k_layout = QVBoxLayout(k_group)
        k_layout.setSpacing(4)

        self._entries['K'] = {}
        k_params = [
            ('fx', 'fx (focal x):'),
            ('fy', 'fy (focal y):'),
            ('cx', 'cx (principal x):'),
            ('cy', 'cy (principal y):'),
        ]

        for key, label_text in k_params:
            row_layout = QHBoxLayout()
            lbl = QLabel(label_text)
            lbl.setFixedWidth(100)
            entry = QLineEdit()
            entry.setFixedWidth(100)
            entry.setText('0.0')
            entry.editingFinished.connect(self._on_param_change)
            self._entries['K'][key] = entry
            row_layout.addWidget(lbl)
            row_layout.addWidget(entry)
            k_layout.addLayout(row_layout)

        parent_layout.addWidget(k_group)

    def _create_distortion_section(self, parent_layout: QVBoxLayout):
        """Create distortion coefficients input section."""
        dist_group = QGroupBox("Distortion Coefficients")
        dist_layout = QVBoxLayout(dist_group)
        dist_layout.setSpacing(4)

        dist_labels = ['k1', 'k2', 'p1', 'p2', 'k3']
        self._dist_entries = {}

        for label in dist_labels:
            row_layout = QHBoxLayout()
            lbl = QLabel(f"{label}:")
            lbl.setFixedWidth(35)
            entry = QLineEdit()
            entry.setFixedWidth(100)
            entry.setText('0.0')
            entry.editingFinished.connect(self._on_param_change)
            self._dist_entries[label] = entry
            row_layout.addWidget(lbl)
            row_layout.addWidget(entry)
            dist_layout.addLayout(row_layout)

        parent_layout.addWidget(dist_group)

    def _create_extrinsic_section(self, parent_layout: QVBoxLayout):
        """Create rotation matrix and translation vector section."""
        ext_group = QGroupBox("Extrinsic Parameters")
        ext_layout = QVBoxLayout(ext_group)

        # Rotation matrix
        r_group = QGroupBox("Rotation Matrix R")
        r_layout = QGridLayout(r_group)
        r_layout.setSpacing(4)

        self._entries['R'] = []
        r_defaults = [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0]
        ]

        for i in range(3):
            row_entries = []
            for j in range(3):
                entry = QLineEdit()
                entry.setFixedWidth(70)
                entry.setText(str(r_defaults[i][j]))
                entry.editingFinished.connect(self._on_param_change)
                row_entries.append(entry)
                r_layout.addWidget(entry, i, j)
            self._entries['R'].append(row_entries)

        ext_layout.addWidget(r_group)

        # Translation vector
        t_group = QGroupBox("Translation Vector T")
        t_layout = QVBoxLayout(t_group)
        t_layout.setSpacing(4)

        t_labels = ['Tx', 'Ty', 'Tz']
        self._t_entries = []

        for label in t_labels:
            row_layout = QHBoxLayout()
            lbl = QLabel(f"{label}:")
            lbl.setFixedWidth(35)
            entry = QLineEdit()
            entry.setFixedWidth(100)
            entry.setText('0.0')
            entry.editingFinished.connect(self._on_param_change)
            self._t_entries.append(entry)
            row_layout.addWidget(lbl)
            row_layout.addWidget(entry)
            t_layout.addLayout(row_layout)

        ext_layout.addWidget(t_group)
        parent_layout.addWidget(ext_group)

    def _on_param_change(self):
        """Handle parameter change events."""
        if self._on_change:
            self._on_change()
        self.param_changed.emit()

    def get_K(self) -> np.ndarray:
        """Get intrinsic matrix as numpy array."""
        K = np.eye(3, dtype=np.float64)
        try:
            fx = float(self._entries['K']['fx'].text().strip())
            fy = float(self._entries['K']['fy'].text().strip())
            cx = float(self._entries['K']['cx'].text().strip())
            cy = float(self._entries['K']['cy'].text().strip())
            K[0, 0] = fx
            K[1, 1] = fy
            K[0, 2] = cx
            K[1, 2] = cy
        except ValueError:
            pass
        return K

    def get_distortion(self) -> np.ndarray:
        """Get distortion coefficients as numpy array."""
        dist = np.zeros(5, dtype=np.float64)
        keys = ['k1', 'k2', 'p1', 'p2', 'k3']
        try:
            for i, key in enumerate(keys):
                val = self._dist_entries[key].text().strip()
                if val:
                    dist[i] = float(val)
        except ValueError:
            pass
        return dist

    def get_R(self) -> np.ndarray:
        """Get rotation matrix as numpy array."""
        R = np.eye(3, dtype=np.float64)
        try:
            for i in range(3):
                for j in range(3):
                    val = self._entries['R'][i][j].text().strip()
                    if val:
                        R[i, j] = float(val)
        except ValueError:
            pass
        return R

    def get_T(self) -> np.ndarray:
        """Get translation vector as numpy array."""
        T = np.zeros(3, dtype=np.float64)
        try:
            for i, entry in enumerate(self._t_entries):
                val = entry.text().strip()
                if val:
                    T[i] = float(val)
        except ValueError:
            pass
        return T

    def set_K(self, K: np.ndarray):
        """Set intrinsic matrix values."""
        self._entries['K']['fx'].setText(f'{K[0, 0]:.6f}')
        self._entries['K']['fy'].setText(f'{K[1, 1]:.6f}')
        self._entries['K']['cx'].setText(f'{K[0, 2]:.6f}')
        self._entries['K']['cy'].setText(f'{K[1, 2]:.6f}')

    def set_distortion(self, dist: np.ndarray):
        """Set distortion coefficients."""
        keys = ['k1', 'k2', 'p1', 'p2', 'k3']
        for i, key in enumerate(keys):
            self._dist_entries[key].setText(f'{dist[i]:.6f}' if i < len(dist) else '0.000000')

    def set_R(self, R: np.ndarray):
        """Set rotation matrix values."""
        for i in range(3):
            for j in range(3):
                self._entries['R'][i][j].setText(f'{R[i, j]:.6f}')

    def set_T(self, T: np.ndarray):
        """Set translation vector values."""
        for i, entry in enumerate(self._t_entries):
            entry.setText(f'{T[i]:.6f}' if i < len(T) else '0.000000')

    def reset_to_identity(self):
        """Reset all parameters to default values."""
        self._entries['K']['fx'].setText('1.0')
        self._entries['K']['fy'].setText('1.0')
        self._entries['K']['cx'].setText('0.0')
        self._entries['K']['cy'].setText('0.0')

        for key in self._dist_entries:
            self._dist_entries[key].setText('0.0')

        defaults_r = [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0]
        ]
        for i in range(3):
            for j in range(3):
                self._entries['R'][i][j].setText(str(defaults_r[i][j]))

        for entry in self._t_entries:
            entry.setText('0.0')

        if self._on_change:
            self._on_change()
        self.param_changed.emit()