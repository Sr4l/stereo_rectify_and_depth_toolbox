import sys
import numpy as np
import cv2
from PIL import Image
from typing import Optional

from PySide6.QtWidgets import (
    QWidget, QGroupBox, QVBoxLayout, QHBoxLayout, QGraphicsView,
    QGraphicsScene, QGraphicsPixmapItem, QLabel, QSlider, QPushButton,
    QMenu, QApplication, QGraphicsTextItem
)
from PySide6.QtCore import Qt, QRectF, QPointF, QPoint
from PySide6.QtGui import QImage, QPixmap, QCursor, QFont, QColor, QPainter


def numpy_to_qimage(arr: np.ndarray) -> QImage:
    """Convert numpy array (BGR or GRAY) to QImage."""
    if arr is None:
        return QImage()
    if len(arr.shape) == 3:
        if arr.shape[2] == 3:
            img_rgb = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
        elif arr.shape[2] == 4:
            img_rgb = cv2.cvtColor(arr, cv2.COLOR_BGRA2RGBA)
        else:
            img_rgb = arr
        return QImage(img_rgb.data, img_rgb.shape[1], img_rgb.shape[0],
                      img_rgb.strides[0], QImage.Format_RGB888)
    else:
        img_gray = cv2.cvtColor(arr, cv2.COLOR_GRAY2RGB)
        return QImage(img_gray.data, img_gray.shape[1], img_gray.shape[0],
                      img_gray.strides[0], QImage.Format_RGB888)


class ImagePanel(QWidget):
    """Panel for displaying images with zoom and pan support using QGraphicsView."""

    def __init__(
        self,
        parent: QWidget = None,
        title: str = "Image",
        show_controls: bool = True,
        value_callback=None,
    ):
        super().__init__(parent)
        self._image: np.ndarray = None
        self._zoom_factor: float = 1.0
        self._pan_offset_x: float = 0.0
        self._pan_offset_y: float = 0.0
        self._drag_start_x: float = 0.0
        self._drag_start_y: float = 0.0
        self._is_dragging: bool = False
        self._value_callback = value_callback
        self._tooltip_item: Optional[QGraphicsTextItem] = None
        self._graphics_view: Optional[QGraphicsView] = None
        self._scene: Optional[QGraphicsScene] = None
        self._pixmap_item: Optional[QGraphicsPixmapItem] = None
        self._zoom_label: Optional[QLabel] = None
        self._control_bar_widget: Optional[QWidget] = None
        self._zoom_value_label: Optional[QLabel] = None
        self._fit_button: Optional[QPushButton] = None
        self._one_to_one_button: Optional[QPushButton] = None
        self._save_button: Optional[QPushButton] = None
        self._zoom_slider: Optional[QSlider] = None

        self._create_widgets()
        # No per-widget stylesheet needed — global theme handles all styling

        group_box = QGroupBox(title)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)

        layout.addWidget(self._graphics_view)
        if show_controls:
            layout.addWidget(self._control_bar_widget)

    def _create_widgets(self):
        """Create the QGraphicsView and control bar."""
        # Graphics view for image display
        self._graphics_view = QGraphicsView()
        self._graphics_view.setRenderHint(QPainter.RenderHint.Antialiasing)
        self._graphics_view.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        # Background color is now handled by global theme stylesheet
        self._graphics_view.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self._graphics_view.setMouseTracking(True)
        self._graphics_view.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._graphics_view.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        self._scene = QGraphicsScene()
        self._graphics_view.setScene(self._scene)

        self._pixmap_item = None

        # Connect events
        self._graphics_view.viewport().installEventFilter(self)

        # Control bar
        self._control_bar_widget = QWidget()
        control_layout = QVBoxLayout(self._control_bar_widget)
        control_layout.setContentsMargins(5, 2, 5, 2)
        control_layout.setSpacing(2)

        self._zoom_label = QLabel("Zoom:")
        self._zoom_value_label = QLabel("100%")
        self._zoom_value_label.setFixedWidth(45)

        self._zoom_slider = QSlider(Qt.Orientation.Horizontal)
        self._zoom_slider.setRange(10, 500)
        self._zoom_slider.setValue(100)
        self._zoom_slider.setFixedWidth(150)
        self._zoom_slider.valueChanged.connect(self._on_zoom_slider_changed)

        self._fit_button = QPushButton("Fit")
        self._fit_button.clicked.connect(self.fit_to_window)

        self._one_to_one_button = QPushButton("1:1")
        self._one_to_one_button.clicked.connect(self.reset_zoom)

        self._save_button = QPushButton("Save image")
        self._save_button.clicked.connect(self.save_image)

        # Row 1: Zoom controls + Fit + 1:1 buttons
        zoom_row = QHBoxLayout()
        zoom_row.addWidget(self._zoom_label)
        zoom_row.addWidget(self._zoom_value_label)
        zoom_row.addWidget(self._zoom_slider)
        zoom_row.addWidget(self._fit_button)
        zoom_row.addWidget(self._one_to_one_button)
        zoom_row.addStretch()
        control_layout.addLayout(zoom_row)

        # Row 2: Save button only
        save_row = QHBoxLayout()
        save_row.addStretch()
        save_row.addWidget(self._save_button)
        save_row.addStretch()
        control_layout.addLayout(save_row)

    def eventFilter(self, obj, event):
        """Event filter for QGraphicsView viewport to handle mouse events."""
        if self._graphics_view is None:
            return super().eventFilter(obj, event)
        if obj is self._graphics_view.viewport():
            if event.type() == event.Type.MouseButtonPress:
                if event.button() == Qt.MouseButton.RightButton:
                    self._show_context_menu(event.position().toPoint())
                    return True
                elif event.button() == Qt.MouseButton.LeftButton:
                    self._drag_start_x = event.position().x() - self._pan_offset_x
                    self._drag_start_y = event.position().y() - self._pan_offset_y
                    self._is_dragging = True
                    self._graphics_view.setDragMode(QGraphicsView.DragMode.NoDrag)
                    return True
            elif event.type() == event.Type.MouseButtonRelease:
                if event.button() == Qt.MouseButton.LeftButton:
                    self._is_dragging = False
                    self._graphics_view.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
                    return True
            elif event.type() == event.Type.MouseMove:
                if self._is_dragging:
                    self._pan_offset_x = event.position().x() - self._drag_start_x
                    self._pan_offset_y = event.position().y() - self._drag_start_y
                    self._update_display()
                    return True
                self._update_tooltip(event.position().toPoint(), event.globalPosition().toPoint())
                return True
            elif event.type() == event.Type.Wheel:
                self._handle_wheel(event.position().toPoint(), event.angleDelta())
                return True

        return super().eventFilter(obj, event)

    def _handle_wheel(self, pos: QPoint, angle_delta: QPoint):
        """Handle mouse wheel zoom."""
        if self._image is None:
            return
        delta = angle_delta.y()
        if delta > 0:
            self._zoom_factor *= 1.1
        else:
            self._zoom_factor /= 1.1
        self._zoom_factor = max(0.1, min(10.0, self._zoom_factor))
        if self._zoom_slider:
            self._zoom_slider.setValue(int(self._zoom_factor * 100))
        self._update_display()

    def _on_zoom_slider_changed(self, value: int):
        """Handle zoom slider change."""
        self._zoom_factor = value / 100.0
        self._update_display()

    def _show_context_menu(self, pos: QPoint):
        """Show context menu on right-click."""
        menu = QMenu(self)

        save_action = menu.addAction("Save Image")
        reset_action = menu.addAction("Reset View")
        menu.addSeparator()
        copy_action = menu.addAction("Copy to Clipboard")

        action = menu.exec(self._graphics_view.mapToGlobal(pos))
        if action is save_action:
            self.save_image()
        elif action is reset_action:
            self.reset_view()
        elif action is copy_action:
            self._copy_to_clipboard()

    def fit_to_window(self):
        """Fit image to viewport size."""
        if self._image is None:
            return
        if self._graphics_view is None:
            return
        viewport = self._graphics_view.viewport()
        vw = viewport.width()
        vh = viewport.height()
        if vw <= 1 or vh <= 1:
            return
        img_h, img_w = self._image.shape[:2]
        scale_w = vw / img_w
        scale_h = vh / img_h
        self._zoom_factor = min(scale_w, scale_h)
        self._pan_offset_x = 0.0
        self._pan_offset_y = 0.0
        if self._zoom_slider:
            self._zoom_slider.setValue(int(self._zoom_factor * 100))
        self._update_display()

    def reset_zoom(self):
        """Reset zoom to 1:1."""
        self._zoom_factor = 1.0
        self._pan_offset_x = 0.0
        self._pan_offset_y = 0.0
        if self._zoom_slider:
            self._zoom_slider.setValue(100)
        self._update_display()

    def reset_view(self):
        """Reset to fit-to-window view."""
        self.fit_to_window()

    def _update_display(self):
        """Update the graphics scene with the current image."""
        if self._scene is None:
            return
        self._scene.clear()
        # After clear, all QGraphicsItems are deleted, so reset tooltip reference
        self._tooltip_item = None
        if self._image is None:
            return

        img_h, img_w = self._image.shape[:2]
        new_w = int(img_w * self._zoom_factor)
        new_h = int(img_h * self._zoom_factor)

        qimg = numpy_to_qimage(self._image)
        pixmap = QPixmap.fromImage(qimg)
        if new_w > 0 and new_h > 0:
            pixmap = pixmap.scaled(new_w, new_h, Qt.AspectRatioMode.KeepAspectRatio,
                                    Qt.TransformationMode.SmoothTransformation)

        self._pixmap_item = self._scene.addPixmap(pixmap)

        if self._graphics_view is None:
            return
        viewport = self._graphics_view.viewport()
        vw = viewport.width()
        vh = viewport.height()
        if vw <= 0 or vh <= 0:
            return

        # Check if _pixmap_item is still valid (not deleted)
        if self._pixmap_item is None or not hasattr(self, '_image') or self._image is None:
            return
        try:
            item_rect = self._pixmap_item.boundingRect()
        except RuntimeError:
            return
        center_x = vw / 2 + self._pan_offset_x
        center_y = vh / 2 + self._pan_offset_y
        self._pixmap_item.setPos(
            center_x - item_rect.width() / 2,
            center_y - item_rect.height() / 2
        )

        if self._zoom_value_label:
            self._zoom_value_label.setText(f"{int(self._zoom_factor * 100)}%")

    def _update_tooltip(self, local_pos: QPoint, global_pos: QPoint):
        """Update tooltip showing pixel coordinates and values."""
        if self._image is None:
            if self._tooltip_item is not None:
                self._tooltip_item.setPlainText("")
            return

        if self._pixmap_item is None:
            return

        if self._graphics_view is None:
            return
        viewport = self._graphics_view.viewport()
        vw = viewport.width()
        vh = viewport.height()
        if vw <= 0 or vh <= 0:
            return

        # Check if _pixmap_item is still valid (not deleted)
        try:
            item_rect = self._pixmap_item.boundingRect()
        except RuntimeError:
            return
        center_x = vw / 2 + self._pan_offset_x
        center_y = vh / 2 + self._pan_offset_y

        item_x = local_pos.x() - (center_x - item_rect.width() / 2)
        item_y = local_pos.y() - (center_y - item_rect.height() / 2)

        if item_rect.width() > 0 and item_rect.height() > 0:
            scale_x = int(self._image.shape[1]) / item_rect.width()
            scale_y = int(self._image.shape[0]) / item_rect.height()
            img_x = int(item_x * scale_x)
            img_y = int(item_y * scale_y)

            if 0 <= img_x < self._image.shape[1] and 0 <= img_y < self._image.shape[0]:
                if self._value_callback is not None:
                    value_str = self._value_callback(img_x, img_y)
                    if value_str:
                        self._show_tooltip(value_str, global_pos)
                        return

                pixel = self._image[img_y, img_x]
                if len(self._image.shape) == 3:
                    value_str = f"({img_x}, {img_y})  R:{pixel[2]:3d}  G:{pixel[1]:3d}  B:{pixel[0]:3d}"
                else:
                    value_str = f"({img_x}, {img_y})  Value:{pixel:3d}"
                self._show_tooltip(value_str, global_pos)
                return

        self._hide_tooltip()

    def _show_tooltip(self, text: str, global_pos: QPoint):
        """Show tooltip at the specified global position."""
        self._hide_tooltip()
        self._tooltip_item = QGraphicsTextItem()
        self._tooltip_item.setDefaultTextColor(QColor(0, 0, 0))
        font = QFont("Arial", 9)
        self._tooltip_item.setFont(font)
        try:
            self._tooltip_item.setHtml(f"<span style='background-color: white; color: black; padding: 2px'>{text}</span>")
        except RuntimeError:
            self._tooltip_item = None
            return
        if self._scene:
            try:
                self._scene.addItem(self._tooltip_item)
            except RuntimeError:
                return

        if self._graphics_view:
            viewport_pos = self._graphics_view.mapFromGlobal(global_pos)
            scene_pos = self._graphics_view.mapToScene(viewport_pos)
            self._tooltip_item.setPos(
                scene_pos.x() + 10,
                scene_pos.y() + 10
            )

    def _hide_tooltip(self):
        """Hide the tooltip."""
        if self._tooltip_item is not None:
            try:
                if self._scene:
                    self._scene.removeItem(self._tooltip_item)
            except RuntimeError:
                pass
            self._tooltip_item = None

    def _copy_to_clipboard(self):
        """Copy current image to clipboard."""
        if self._image is None:
            return
        try:
            qimg = numpy_to_qimage(self._image)
            QApplication.clipboard().setImage(qimg)
        except Exception:
            pass

    def save_image(self):
        """Save current image to file."""
        if self._image is None:
            return
        from PySide6.QtWidgets import QFileDialog
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Image",
            "",
            "PNG files (*.png);;JPEG files (*.jpg);;All files (*)",
        )
        if file_path:
            try:
                qimg = numpy_to_qimage(self._image)
                qimg.save(file_path)
            except Exception as e:
                from PySide6.QtWidgets import QMessageBox
                QMessageBox.critical(self, "Error", f"Failed to save image: {str(e)}")

    def set_image(self, image: np.ndarray):
        """Set the image to display."""
        self._image = image
        self.fit_to_window()

    def clear(self):
        """Clear the displayed image."""
        self._image = None
        self._update_display()


class ThumbnailPanel(QWidget):
    """Small thumbnail image panel for image preview."""

    def __init__(
        self,
        parent: QWidget = None,
        title: str = "Preview",
        size: tuple = (200, 150),
        load_button_text: str = "Load Image",
    ):
        super().__init__(parent)
        self._image: Optional[np.ndarray] = None
        self._size = size
        self._title = title
        self._graphics_view: Optional[QGraphicsView] = None
        self._scene: Optional[QGraphicsScene] = None
        self._pixmap_item: Optional[QGraphicsPixmapItem] = None
        self._load_button: Optional[QPushButton] = None

        # No per-widget stylesheet needed — global theme handles all styling
        self._create_widgets(load_button_text)

    def _create_widgets(self, load_button_text: str):
        """Create the thumbnail widget hierarchy."""
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)

        self._group_box = QGroupBox(self._title)
        # Set bold font on the group box title
        font = QFont()
        font.setBold(True)
        self._group_box.setFont(font)
        group_layout = QVBoxLayout(self._group_box)
        group_layout.setContentsMargins(5, 5, 5, 5)

        # Load button at the top
        self._load_button = QPushButton(load_button_text)
        self._load_button.setStyleSheet("""
            QPushButton {
                background-color: rgba(14, 99, 156, 180);
                color: white;
                border: none;
                padding: 8px 12px;
                border-radius: 3px;
                font-weight: bold;
                font-size: 12px;
            }
            QPushButton:hover {
                background-color: rgba(17, 119, 187, 200);
            }
        """)
        group_layout.addWidget(self._load_button)

        self._graphics_view = QGraphicsView()
        self._graphics_view.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        # Background color is now handled by global theme stylesheet
        self._graphics_view.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._graphics_view.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._graphics_view.setMinimumHeight(150)

        self._scene = QGraphicsScene()
        self._graphics_view.setScene(self._scene)
        self._pixmap_item = None

        group_layout.addWidget(self._graphics_view)
        main_layout.addWidget(self._group_box)

    def get_load_button(self) -> Optional[QPushButton]:
        """Return the load button so the main window can connect signals."""
        return self._load_button

    def set_image(self, image: np.ndarray):
        """Set the thumbnail image."""
        self._image = image
        if self._load_button:
            self._load_button.setVisible(image is not None)
        self._update_display()

    def _update_display(self):
        """Update thumbnail display."""
        if self._scene is None:
            return
        self._scene.clear()
        if self._image is None:
            label = QGraphicsTextItem("No Image")
            label.setDefaultTextColor(QColor(102, 102, 102))
            label.setFont(QFont("Arial", 10))
            self._scene.addItem(label)
            return

        img_h, img_w = self._image.shape[:2]
        qimg = numpy_to_qimage(self._image)
        pixmap = QPixmap.fromImage(qimg)

        if self._graphics_view is None:
            self._pixmap_item = self._scene.addPixmap(pixmap)
            return

        viewport = self._graphics_view.viewport()
        vw = viewport.width()
        if vw > 0:
            pixmap = pixmap.scaledToWidth(vw, Qt.TransformationMode.SmoothTransformation)
            self._scene.setSceneRect(0, 0, vw, self._graphics_view.viewport().height())

        self._pixmap_item = self._scene.addPixmap(pixmap)

    def clear(self):
        """Clear the thumbnail."""
        self._image = None
        self._update_display()