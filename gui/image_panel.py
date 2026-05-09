import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import numpy as np
from typing import Optional, Callable
import cv2


class ImagePanel(ttk.LabelFrame):
    """Panel for displaying images with zoom and pan support."""
    
    def __init__(
        self,
        parent,
        title: str = "Image",
        show_controls: bool = True,
        **kwargs
    ):
        super().__init__(parent, text=title, **kwargs)
        
        self.image: Optional[np.ndarray] = None
        self.photo_image: Optional[ImageTk.PhotoImage] = None
        self.zoom = 1.0
        self.pan_x = 0
        self.pan_y = 0
        self.drag_start_x = 0
        self.drag_start_y = 0
        
        self._create_widgets(show_controls)
    
    def _create_widgets(self, show_controls: bool):
        """Create image display and control widgets."""
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)
        
        self.canvas = tk.Canvas(
            self,
            bg='#2b2b2b',
            highlightthickness=0
        )
        self.canvas.grid(row=0, column=0, sticky='nsew', padx=2, pady=2)
        
        self.canvas.bind('<Configure>', self._on_canvas_resize)
        self.canvas.bind('<MouseWheel>', self._on_mouse_wheel)
        self.canvas.bind('<ButtonPress-1>', self._on_drag_start)
        self.canvas.bind('<B1-Motion>', self._on_drag_motion)
        self.canvas.bind('<Button-3>', self._show_context_menu)
        self.canvas.bind('<Motion>', self._on_mouse_motion)
        
        self.tooltip = None
        self.tooltip_label = None
        
        if show_controls:
            self._create_control_bar()
    
    def _create_control_bar(self):
        """Create zoom and action control bar."""
        control_frame = ttk.Frame(self)
        control_frame.grid(row=1, column=0, sticky='ew', padx=2, pady=2)
        
        ttk.Label(control_frame, text="Zoom:").pack(side=tk.LEFT, padx=2)
        
        self.zoom_var = tk.StringVar(value="100%")
        zoom_label = ttk.Label(control_frame, textvariable=self.zoom_var, width=6)
        zoom_label.pack(side=tk.LEFT, padx=2)
        
        zoom_slider = ttk.Scale(
            control_frame,
            from_=0.1,
            to=5.0,
            orient=tk.HORIZONTAL,
            length=150,
            command=self._on_zoom_change
        )
        zoom_slider.set(1.0)
        zoom_slider.pack(side=tk.LEFT, padx=5)
        
        ttk.Button(
            control_frame,
            text="Fit",
            command=self._fit_to_window
        ).pack(side=tk.LEFT, padx=2)
        
        ttk.Button(
            control_frame,
            text="1:1",
            command=self._reset_zoom
        ).pack(side=tk.LEFT, padx=2)
        
        ttk.Button(
            control_frame,
            text="Save",
            command=self._save_image
        ).pack(side=tk.LEFT, padx=2)
    
    def _on_canvas_resize(self, event):
        """Handle canvas resize event."""
        self._update_display()
    
    def _on_mouse_wheel(self, event):
        """Handle mouse wheel zoom."""
        if event.delta > 0:
            self.zoom *= 1.1
        else:
            self.zoom /= 1.1
        
        self.zoom = max(0.1, min(10.0, self.zoom))
        self._update_display()
    
    def _on_zoom_change(self, value):
        """Handle zoom slider change."""
        self.zoom = float(value)
        if hasattr(self, 'zoom_var') and self.zoom_var is not None:
            self.zoom_var.set(f"{int(self.zoom * 100)}%")
        self._update_display()
    
    def _on_drag_start(self, event):
        """Handle drag start event."""
        self.drag_start_x = event.x - self.pan_x
        self.drag_start_y = event.y - self.pan_y
    
    def _on_drag_motion(self, event):
        """Handle drag motion event."""
        self.pan_x = event.x - self.drag_start_x
        self.pan_y = event.y - self.drag_start_y
        self._update_display()
    
    def _show_context_menu(self, event):
        """Show context menu on right-click."""
        menu = tk.Menu(self, tearoff=0)
        menu.add_command(label="Save Image", command=self._save_image)
        menu.add_command(label="Reset View", command=self._reset_view)
        menu.add_separator()
        menu.add_command(label="Copy to Clipboard", command=self._copy_to_clipboard)
        
        try:
            menu.tk_popup(event.x_root, event.y_root)
        finally:
            menu.grab_release()
    
    def _fit_to_window(self):
        """Fit image to canvas size."""
        if self.image is None:
            return
        
        canvas_w = self.canvas.winfo_width()
        canvas_h = self.canvas.winfo_height()
        
        if canvas_w <= 1 or canvas_h <= 1:
            return
        
        img_h, img_w = self.image.shape[:2]
        
        scale_w = canvas_w / img_w
        scale_h = canvas_h / img_h
        
        self.zoom = min(scale_w, scale_h)
        self.pan_x = 0
        self.pan_y = 0
        
        self._update_display()
    
    def _reset_zoom(self):
        """Reset zoom to 1:1."""
        self.zoom = 1.0
        self.pan_x = 0
        self.pan_y = 0
        self._update_display()
    
    def _reset_view(self):
        """Reset zoom and pan to default."""
        self._fit_to_window()
    
    def _update_display(self):
        """Update the canvas with the current image."""
        if self.image is None:
            self.canvas.delete('all')
            return
        
        canvas_w = self.canvas.winfo_width()
        canvas_h = self.canvas.winfo_height()
        
        if canvas_w <= 1 or canvas_h <= 1:
            return
        
        img_h, img_w = self.image.shape[:2]
        
        new_w = int(img_w * self.zoom)
        new_h = int(img_h * self.zoom)
        
        if len(self.image.shape) == 3:
            img_rgb = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        else:
            img_rgb = cv2.cvtColor(self.image, cv2.COLOR_GRAY2RGB)
        
        img_pil = Image.fromarray(img_rgb)
        img_resized = img_pil.resize((new_w, new_h), Image.Resampling.LANCZOS)
        
        self.photo_image = ImageTk.PhotoImage(img_resized)
        
        self.canvas.delete('all')
        
        x = canvas_w // 2 + self.pan_x
        y = canvas_h // 2 + self.pan_y
        
        self.canvas.create_image(
            x, y,
            anchor=tk.CENTER,
            image=self.photo_image
        )
        
        if hasattr(self, 'zoom_var') and self.zoom_var is not None:
            self.zoom_var.set(f"{int(self.zoom * 100)}%")
    
    def _save_image(self):
        """Save current image to file."""
        if self.image is None:
            return
        
        from tkinter import filedialog
        file_path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[
                ("PNG files", "*.png"),
                ("JPEG files", "*.jpg"),
                ("All files", "*.*")
            ]
        )
        
        if file_path:
            try:
                if len(self.image.shape) == 3:
                    img_save = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
                else:
                    img_save = self.image
                
                cv2.imwrite(file_path, img_save)
            except Exception as e:
                tk.messagebox.showerror("Error", f"Failed to save image: {e}")
    
    def _on_mouse_motion(self, event):
        """Handle mouse motion to show pixel value tooltip."""
        if self.image is None:
            self._hide_tooltip()
            return
        
        canvas_w = self.canvas.winfo_width()
        canvas_h = self.canvas.winfo_height()
        
        if canvas_w <= 1 or canvas_h <= 1:
            self._hide_tooltip()
            return
        
        img_h, img_w = self.image.shape[:2]
        new_w = int(img_w * self.zoom)
        new_h = int(img_h * self.zoom)
        
        if new_w <= 0 or new_h <= 0:
            self._hide_tooltip()
            return
        
        x = canvas_w // 2 + self.pan_x
        y = canvas_h // 2 + self.pan_y
        
        img_x = int((event.x - x) / self.zoom + img_w // 2)
        img_y = int((event.y - y) / self.zoom + img_h // 2)
        
        if 0 <= img_x < img_w and 0 <= img_y < img_h:
            pixel = self.image[img_y, img_x]
            
            if len(self.image.shape) == 3:
                value_str = f"({img_x}, {img_y})  R:{pixel[2]:3d}  G:{pixel[1]:3d}  B:{pixel[0]:3d}"
            else:
                value_str = f"({img_x}, {img_y})  Value:{pixel:3d}"
            
            self._show_tooltip(value_str, event.x_root, event.y_root)
        else:
            self._hide_tooltip()
    
    def _show_tooltip(self, text: str, x: int, y: int):
        """Show tooltip at specified position."""
        if self.tooltip_label is not None:
            self.tooltip_label.destroy()
            self.tooltip_label = None
        
        self.tooltip_label = tk.Toplevel(self)
        self.tooltip_label.wm_overrideredirect(True)
        self.tooltip_label.wm_attributes("-topmost", True)
        
        label = tk.Label(
            self.tooltip_label,
            text=text,
            background="#ffffe0",
            foreground="#000000",
            font=('Arial', 9),
            padx=4,
            pady=2,
            relief='solid',
            borderwidth=1
        )
        label.pack()
        
        self.tooltip_label.wm_geometry(f"+{x + 10}+{y + 10}")
    
    def _hide_tooltip(self):
        """Hide the tooltip."""
        if self.tooltip_label is not None:
            try:
                self.tooltip_label.destroy()
            except Exception:
                pass
            self.tooltip_label = None
    
    def _copy_to_clipboard(self):
        """Copy current image to clipboard."""
        if self.image is None:
            return
        
        try:
            img_pil = Image.fromarray(cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB))
            img_pil.save('clipboard_temp.png')
            self.clipboard_clear()
            self.clipboard_append(file_path='clipboard_temp.png')
        except Exception:
            pass
    
    def set_image(self, image: Optional[np.ndarray]):
        """Set the image to display."""
        self.image = image
        self._fit_to_window()
    
    def clear(self):
        """Clear the displayed image."""
        self.image = None
        self.photo_image = None
        self.canvas.delete('all')


class ThumbnailPanel(ttk.LabelFrame):
    """Small thumbnail image panel for image preview."""
    
    def __init__(
        self,
        parent,
        title: str = "Preview",
        size: tuple = (200, 150),
        **kwargs
    ):
        super().__init__(parent, text=title, **kwargs)
        
        self.size = size
        self.image: Optional[np.ndarray] = None
        self.photo_image: Optional[ImageTk.PhotoImage] = None
        
        self._create_widgets()
    
    def _create_widgets(self):
        """Create thumbnail display."""
        self.canvas = tk.Canvas(
            self,
            width=self.size[0],
            height=self.size[1],
            bg='#2b2b2b',
            highlightthickness=0
        )
        self.canvas.pack(fill=tk.BOTH, expand=True, padx=2, pady=2)
    
    def set_image(self, image: Optional[np.ndarray]):
        """Set the thumbnail image."""
        self.image = image
        self._update_display()
    
    def _update_display(self):
        """Update thumbnail display."""
        if self.image is None:
            self.canvas.delete('all')
            self.canvas.create_text(
                self.size[0] // 2,
                self.size[1] // 2,
                text="No Image",
                fill='#666666',
                font=('Arial', 10)
            )
            return
        
        h, w = self.image.shape[:2]
        
        scale = min(self.size[0] / w, self.size[1] / h)
        new_w = int(w * scale)
        new_h = int(h * scale)
        
        if len(self.image.shape) == 3:
            img_rgb = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        else:
            img_rgb = cv2.cvtColor(self.image, cv2.COLOR_GRAY2RGB)
        
        img_pil = Image.fromarray(img_rgb)
        img_resized = img_pil.resize((new_w, new_h), Image.Resampling.LANCZOS)
        
        self.photo_image = ImageTk.PhotoImage(img_resized)
        
        self.canvas.delete('all')
        self.canvas.create_image(
            self.size[0] // 2,
            self.size[1] // 2,
            anchor=tk.CENTER,
            image=self.photo_image
        )
    
    def clear(self):
        """Clear the thumbnail."""
        self.image = None
        self.photo_image = None
        self._update_display()
