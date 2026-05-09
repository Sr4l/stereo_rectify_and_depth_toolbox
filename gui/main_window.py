import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import numpy as np
import cv2
from PIL import Image
import threading
import json
from typing import Optional
import os

from .param_panel import CameraParamPanel
from .image_panel import ImagePanel, ThumbnailPanel
from core.rectifier import StereoRectifier
from core.depth import DepthEstimator


class StereoCalibrationGUI:
    """Main application window for stereo camera calibration and depth estimation."""
    
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Stereo Camera Calibration & Depth Toolbox")
        self.root.geometry("1600x900")
        self.root.minsize(1200, 700)
        
        self.rectifier = StereoRectifier()
        self.depth_estimator = DepthEstimator()
        
        self.left_image_path: Optional[str] = None
        self.right_image_path: Optional[str] = None
        
        self.update_debounce_id = None
        self.depth_debounce_id = None
        
        self._setup_styles()
        self._create_ui()
        self._bind_events()
    
    def _setup_styles(self):
        """Configure application styles."""
        style = ttk.Style()
        
        try:
            style.theme_use('clam')
        except Exception:
            pass
        
        style.configure('TFrame', background='#1e1e1e')
        style.configure('TLabel', background='#1e1e1e', foreground='#ffffff')
        style.configure('TButton', padding=5)
        style.configure('TLabelframe', background='#1e1e1e', foreground='#ffffff')
        style.configure('TLabelframe.Label', background='#252526', foreground='#007acc', font=('Arial', 10, 'bold'))
    
    def _create_ui(self):
        """Create the user interface."""
        main_container = ttk.Frame(self.root)
        main_container.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        main_container.grid_rowconfigure(0, weight=1)
        main_container.grid_columnconfigure(1, weight=3)
        main_container.grid_columnconfigure(2, weight=3)
        
        self._create_left_panel(main_container)
        self._create_center_panel(main_container)
        self._create_right_panel(main_container)
        
        self._create_status_bar(main_container)
    
    def _create_left_panel(self, parent):
        """Create left sidebar with parameter controls."""
        left_frame = ttk.Frame(parent, width=320)
        left_frame.grid(row=0, column=0, sticky='ns', padx=(0, 5))
        left_frame.grid_propagate(False)
        
        canvas = tk.Canvas(left_frame, bg='#1e1e1e', highlightthickness=0)
        scrollbar = ttk.Scrollbar(left_frame, orient="vertical", command=canvas.yview)
        
        self.params_scroll_frame = ttk.Frame(canvas)
        
        canvas_window = canvas.create_window((0, 0), window=self.params_scroll_frame, anchor="nw")
        
        def on_frame_configure(event):
            canvas.configure(scrollregion=canvas.bbox("all"))
        
        def on_canvas_configure(event):
            canvas.itemconfig(canvas_window, width=event.width)
        
        self.params_scroll_frame.bind("<Configure>", on_frame_configure)
        canvas.bind("<Configure>", on_canvas_configure)
        
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        canvas.bind_all("<MouseWheel>", lambda e: canvas.yview_scroll(int(-1*(e.delta/120)), "units"))
        
        self._create_load_section(self.params_scroll_frame)
        
        ttk.Separator(self.params_scroll_frame, orient='horizontal').pack(fill=tk.X, pady=10)
        
        self.left_param_panel = CameraParamPanel(
            self.params_scroll_frame,
            title="Left Camera Parameters",
            on_change=self._on_param_change
        )
        self.left_param_panel.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Separator(self.params_scroll_frame, orient='horizontal').pack(fill=tk.X, pady=10)
        
        self.right_param_panel = CameraParamPanel(
            self.params_scroll_frame,
            title="Right Camera Parameters",
            on_change=self._on_param_change
        )
        self.right_param_panel.pack(fill=tk.X, padx=5, pady=5)
    
    def _create_load_section(self, parent):
        """Create image loading section."""
        frame = ttk.LabelFrame(parent, text="Load Images")
        frame.pack(fill=tk.X, padx=5, pady=5)
        
        left_frame = ttk.Frame(frame)
        left_frame.pack(fill=tk.X, pady=5)
        
        self.btn_load_left = ttk.Button(
            left_frame,
            text="Load Left Image",
            command=self._load_left_image
        )
        self.btn_load_left.pack(side=tk.LEFT, padx=2)
        
        self.left_thumbnail = ThumbnailPanel(left_frame, title="Preview", size=(140, 100))
        self.left_thumbnail.pack(side=tk.RIGHT, padx=5)
        
        right_frame = ttk.Frame(frame)
        right_frame.pack(fill=tk.X, pady=5)
        
        self.btn_load_right = ttk.Button(
            right_frame,
            text="Load Right Image",
            command=self._load_right_image
        )
        self.btn_load_right.pack(side=tk.LEFT, padx=2)
        
        self.right_thumbnail = ThumbnailPanel(right_frame, title="Preview", size=(140, 100))
        self.right_thumbnail.pack(side=tk.RIGHT, padx=5)
        
        btn_frame = ttk.Frame(frame)
        btn_frame.pack(fill=tk.X, pady=5)
        
        ttk.Button(
            btn_frame,
            text="Save Calibration",
            command=self._save_calibration
        ).pack(side=tk.LEFT, padx=2)
        
        ttk.Button(
            btn_frame,
            text="Load Calibration",
            command=self._load_calibration
        ).pack(side=tk.LEFT, padx=2)
    
    def _create_center_panel(self, parent):
        """Create center panel with rectified images."""
        center_frame = ttk.Frame(parent)
        center_frame.grid(row=0, column=1, sticky='nsew', padx=(0, 5))
        
        center_frame.grid_rowconfigure(0, weight=2)
        center_frame.grid_rowconfigure(1, weight=3)
        center_frame.grid_columnconfigure(0, weight=1)
        center_frame.grid_columnconfigure(1, weight=1)
        
        rectified_frame = ttk.LabelFrame(center_frame, text="Rectified Views")
        rectified_frame.grid(row=0, column=0, columnspan=2, sticky='nsew', padx=2, pady=2)
        
        rectified_frame.grid_rowconfigure(0, weight=1)
        rectified_frame.grid_columnconfigure(0, weight=1)
        rectified_frame.grid_columnconfigure(1, weight=1)
        
        self.rectified_left_panel = ImagePanel(rectified_frame, "Left Rectified", show_controls=False)
        self.rectified_left_panel.grid(row=0, column=0, sticky='nsew', padx=2, pady=2)
        
        self.rectified_right_panel = ImagePanel(rectified_frame, "Right Rectified", show_controls=False)
        self.rectified_right_panel.grid(row=0, column=1, sticky='nsew', padx=2, pady=2)
        
        options_frame = ttk.Frame(rectified_frame)
        options_frame.grid(row=1, column=0, columnspan=2, sticky='ew', padx=2, pady=5)
        
        self.epipolar_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            options_frame,
            text="Show Epipolar Lines",
            variable=self.epipolar_var,
            command=self._toggle_epipolar_lines
        ).pack(side=tk.LEFT, padx=5)
        
        ttk.Label(options_frame, text=" ").pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        ttk.Button(
            options_frame,
            text="Save Rectified Images",
            command=self._save_rectified_images
        ).pack(side=tk.LEFT, padx=2)
    
    def _create_right_panel(self, parent):
        """Create right panel with depth map and BM controls."""
        right_frame = ttk.Frame(parent)
        right_frame.grid(row=0, column=2, sticky='nsew')
        
        right_frame.grid_rowconfigure(0, weight=1)
        right_frame.grid_rowconfigure(1, weight=0)
        right_frame.grid_rowconfigure(2, weight=0)
        right_frame.grid_rowconfigure(3, weight=0)
        right_frame.grid_columnconfigure(0, weight=1)
        
        depth_frame = ttk.LabelFrame(right_frame, text="Depth Map / Disparity")
        depth_frame.grid(row=0, column=0, sticky='nsew', padx=2, pady=2)
        
        self.depth_panel = ImagePanel(
            depth_frame, 
            "Depth Visualization", 
            show_controls=True,
            value_callback=self._get_depth_value
        )
        self.depth_panel.pack(fill=tk.BOTH, expand=True, padx=2, pady=2)
        
        bm_frame = ttk.LabelFrame(right_frame, text="StereoBM Parameters")
        bm_frame.grid(row=1, column=0, sticky='ew', padx=2, pady=5)
        
        self._create_bm_controls(bm_frame)
        
        vis_frame = ttk.LabelFrame(right_frame, text="Visualization Controls")
        vis_frame.grid(row=2, column=0, sticky='ew', padx=2, pady=5)
        
        self._create_visualization_controls(vis_frame)
    
    def _create_bm_controls(self, parent):
        """Create StereoBM parameter controls."""
        controls = [
            ('numDisparities', 'Num Disp:', 16, 256, 16, 16),
            ('blockSize', 'Block:', 5, 25, 2, 9),
            ('minDisparity', 'Min Disp:', -100, 100, 1, 0),
            ('uniquenessRatio', 'Unique:', 1, 100, 1, 10),
            ('speckleWindowSize', 'Speckle W:', 0, 200, 1, 100),
            ('speckleRange', 'Speckle R:', 0, 50, 1, 1),
        ]
        
        self.bm_vars = {}
        self.bm_scales = {}
        
        for i, (key, label, from_val, to_val, step, default) in enumerate(controls):
            row = i % 3
            col = (i // 3) * 2
            
            frame = ttk.Frame(parent)
            frame.grid(row=row, column=col, sticky='w', padx=3, pady=2)
            
            ttk.Label(frame, text=label, width=10).pack(side=tk.LEFT)
            
            var = tk.IntVar(value=default)
            self.bm_vars[key] = var
            
            scale = ttk.Scale(
                frame,
                from_=from_val,
                to=to_val,
                orient=tk.HORIZONTAL,
                length=80,
                variable=var,
                command=lambda v, k=key: self._on_bm_param_change(k)
            )
            scale.pack(side=tk.LEFT, padx=2)
            self.bm_scales[key] = scale
            
            value_label = ttk.Label(frame, textvariable=var, width=4)
            value_label.pack(side=tk.LEFT)
    
    def _create_visualization_controls(self, parent):
        """Create visualization control widgets."""
        row1 = ttk.Frame(parent)
        row1.pack(fill=tk.X, padx=5, pady=2)
        
        ttk.Button(
            row1,
            text="Update Depth",
            command=self._update_depth
        ).pack(side=tk.LEFT, padx=2)
        
        ttk.Button(
            row1,
            text="Save Depth Map",
            command=self._save_depth_map
        ).pack(side=tk.LEFT, padx=2)
        
        row2 = ttk.Frame(parent)
        row2.pack(fill=tk.X, padx=5, pady=2)
        
        ttk.Label(row2, text="View:").pack(side=tk.LEFT, padx=2)
        
        self.view_mode_var = tk.StringVar(value="disparity")
        view_combo = ttk.Combobox(
            row2,
            textvariable=self.view_mode_var,
            values=["disparity", "depth (mm)"],
            width=12,
            state="readonly"
        )
        view_combo.pack(side=tk.LEFT, padx=2)
        view_combo.bind('<<ComboboxSelected>>', lambda e: self._update_depth())
        
        ttk.Label(row2, text="Colormap:").pack(side=tk.LEFT, padx=(10, 2))
        
        self.colormap_var = tk.StringVar(value="JET")
        colormap_combo = ttk.Combobox(
            row2,
            textvariable=self.colormap_var,
            values=["JET", "VIRIDIS", "MAGMA", "INFERNO", "PLASMA", "CIVIDIS"],
            width=8,
            state="readonly"
        )
        colormap_combo.pack(side=tk.LEFT, padx=2)
        colormap_combo.bind('<<ComboboxSelected>>', lambda e: self._update_depth())
    
    def _create_status_bar(self, parent):
        """Create status bar at bottom."""
        status_frame = ttk.Frame(parent)
        status_frame.grid(row=1, column=0, columnspan=3, sticky='ew', pady=(5, 0))
        
        self.status_var = tk.StringVar(value="Ready")
        status_label = ttk.Label(status_frame, textvariable=self.status_var, anchor='w')
        status_label.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        self.process_time_var = tk.StringVar(value="")
        time_label = ttk.Label(status_frame, textvariable=self.process_time_var, anchor='e')
        time_label.pack(side=tk.RIGHT, padx=10)
    
    def _bind_events(self):
        """Bind keyboard events."""
        self.root.bind('<Control-s>', lambda e: self._save_rectified_images())
        self.root.bind('<Control-o>', lambda e: self._load_left_image())
        self.root.bind('<Control-r>', lambda e: self._load_right_image())
        self.root.bind('<F5>', lambda e: self._update_rectification())
    
    def _load_left_image(self):
        """Load left camera image."""
        file_path = filedialog.askopenfilename(
            title="Select Left Image",
            filetypes=[
                ("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff *.tif"),
                ("All files", "*.*")
            ]
        )
        
        if file_path:
            self.left_image_path = file_path
            image = cv2.imread(file_path)
            
            if image is not None:
                self.rectifier.set_left_image(image)
                self.left_thumbnail.set_image(image)
                self._update_rectification()
                self.status_var.set(f"Left image loaded: {os.path.basename(file_path)}")
    
    def _load_right_image(self):
        """Load right camera image."""
        file_path = filedialog.askopenfilename(
            title="Select Right Image",
            filetypes=[
                ("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff *.tif"),
                ("All files", "*.*")
            ]
        )
        
        if file_path:
            self.right_image_path = file_path
            image = cv2.imread(file_path)
            
            if image is not None:
                self.rectifier.set_right_image(image)
                self.right_thumbnail.set_image(image)
                self._update_rectification()
                self.status_var.set(f"Right image loaded: {os.path.basename(file_path)}")
    
    def _on_param_change(self):
        """Handle parameter change with debouncing."""
        if self.update_debounce_id:
            self.root.after_cancel(self.update_debounce_id)
        
        self.update_debounce_id = self.root.after(500, self._update_rectification)
    
    def _update_rectification(self):
        """Update rectified images."""
        if self.rectifier.left_image is None or self.rectifier.right_image is None:
            return
        
        try:
            start_time = cv2.getTickCount()
            
            self.rectifier.set_left_intrinsics(
                self.left_param_panel.get_K(),
                self.left_param_panel.get_distortion()
            )
            self.rectifier.set_right_intrinsics(
                self.right_param_panel.get_K(),
                self.right_param_panel.get_distortion()
            )
            self.rectifier.set_extrinsics(
                self.right_param_panel.get_R(),
                self.right_param_panel.get_T()
            )
            
            rect_left, rect_right = self.rectifier.rectify()
            
            if rect_left is not None and rect_right is not None:
                if self.epipolar_var.get():
                    rect_left = self.rectifier.draw_epipolar_lines(rect_left)
                    rect_right = self.rectifier.draw_epipolar_lines(rect_right)
                
                self.rectified_left_panel.set_image(rect_left)
                self.rectified_right_panel.set_image(rect_right)
                
                self._update_depth()
            
            elapsed = (cv2.getTickCount() - start_time) / cv2.getTickFrequency()
            self.process_time_var.set(f"Rectification: {elapsed*1000:.1f}ms")
            self.status_var.set("Rectification updated")
            
        except Exception as e:
            error_msg = str(e)
            if 'stereoRectify' in error_msg or 'nt > 0.0' in error_msg:
                error_msg = "Invalid camera parameters. Please check:\n- Focal lengths (fx, fy) must be positive\n- Principal point (cx, cy) should be within image bounds"
            self.status_var.set(f"Error: {error_msg}")
    
    def _toggle_epipolar_lines(self):
        """Toggle epipolar lines display."""
        self._update_rectification()
    
    def _on_bm_param_change(self, key):
        """Handle BM parameter change with debouncing."""
        if self.depth_debounce_id:
            self.root.after_cancel(self.depth_debounce_id)
        
        self.depth_debounce_id = self.root.after(300, self._update_depth)
    
    def _update_depth(self):
        """Update depth map."""
        if self.rectifier.rectified_left is None or self.rectifier.rectified_right is None:
            return
        
        try:
            start_time = cv2.getTickCount()
            
            params = {key: var.get() for key, var in self.bm_vars.items()}
            
            if 'blockSize' in params:
                if params['blockSize'] % 2 == 0:
                    params['blockSize'] += 1
                params['blockSize'] = max(5, min(255, params['blockSize']))
            
            self.depth_estimator.set_bm_params(**params)
            
            K_left = self.left_param_panel.get_K()
            T_right = self.right_param_panel.get_T()
            baseline = np.linalg.norm(T_right)
            focal_length = (K_left[0, 0] + K_left[1, 1]) / 2.0
            self.depth_estimator.set_camera_params(baseline, focal_length)
            
            disparity = self.depth_estimator.compute_disparity(
                self.rectifier.rectified_left,
                self.rectifier.rectified_right
            )
            
            if disparity is not None:
                colormap_name = self.colormap_var.get()
                colormap_map = {
                    'JET': cv2.COLORMAP_JET,
                    'VIRIDIS': cv2.COLORMAP_VIRIDIS,
                    'MAGMA': cv2.COLORMAP_MAGMA,
                    'INFERNO': cv2.COLORMAP_INFERNO,
                    'PLASMA': cv2.COLORMAP_PLASMA,
                    'CIVIDIS': cv2.COLORMAP_CIVIDIS
                }
                colormap = colormap_map.get(colormap_name, cv2.COLORMAP_JET)
                
                view_mode = self.view_mode_var.get()
                
                if view_mode == "depth (mm)":
                    depth_map = self.depth_estimator.compute_depth()
                    if depth_map is not None:
                        depth_mm = depth_map * 1000
                        depth_mm_clipped = np.clip(depth_mm, 0, 10000)
                        depth_normalized = ((depth_mm_clipped - depth_mm_clipped.min()) / 
                                          (depth_mm_clipped.max() - depth_mm_clipped.min()) * 255).astype(np.uint8)
                        colored_depth = cv2.applyColorMap(depth_normalized, colormap)
                        self.depth_panel.set_image(colored_depth)
                    
                    stats = self.depth_estimator.get_depth_stats()
                    self.status_var.set(
                        f"Depth - Min: {stats['min']*1000:.1f}mm, Max: {stats['max']*1000:.1f}mm, "
                        f"Mean: {stats['mean']*1000:.1f}mm"
                    )
                else:
                    colored_disparity = self.depth_estimator.apply_colormap(disparity, colormap)
                    self.depth_panel.set_image(colored_disparity)
                    
                    stats = self.depth_estimator.get_disparity_stats()
                    self.status_var.set(
                        f"Disparity - Min: {stats['min']:.2f}, Max: {stats['max']:.2f}, "
                        f"Mean: {stats['mean']:.2f}"
                    )
            
            elapsed = (cv2.getTickCount() - start_time) / cv2.getTickFrequency()
            self.process_time_var.set(f"Depth: {elapsed*1000:.1f}ms")
            
        except Exception as e:
            self.status_var.set(f"Depth Error: {str(e)}")
    
    def _get_depth_value(self, x: int, y: int) -> str:
        """Get depth/disparity value at pixel coordinates for tooltip."""
        if self.depth_estimator.disparity is None or y >= self.depth_estimator.disparity.shape[0] or x >= self.depth_estimator.disparity.shape[1]:
            return None
        
        try:
            disp_val = self.depth_estimator.disparity[y, x]
            
            if disp_val <= 0:
                return f"({x}, {y})  Disparity: 0.00  Depth: N/A"
            
            view_mode = self.view_mode_var.get()
            
            if view_mode == "depth (mm)":
                K_left = self.left_param_panel.get_K()
                T_right = self.right_param_panel.get_T()
                baseline = np.linalg.norm(T_right)
                focal_length = (K_left[0, 0] + K_left[1, 1]) / 2.0
                
                depth_m = (baseline * focal_length) / disp_val
                depth_mm = depth_m * 1000
                
                if depth_mm > 10000:
                    return f"({x}, {y})  Disparity: {disp_val:.2f}  Depth: >10000mm"
                
                return f"({x}, {y})  Disparity: {disp_val:.2f}  Depth: {depth_mm:.1f}mm"
            else:
                return f"({x}, {y})  Disparity: {disp_val:.2f}"
                
        except Exception:
            return None
    
    def _save_rectified_images(self):
        """Save rectified images to files."""
        if self.rectifier.rectified_left is None:
            messagebox.showwarning("Warning", "No rectified images to save")
            return
        
        left_path = filedialog.asksaveasfilename(
            title="Save Left Rectified Image",
            defaultextension=".png",
            filetypes=[("PNG files", "*.png"), ("JPEG files", "*.jpg")]
        )
        
        if left_path:
            cv2.imwrite(left_path, self.rectifier.rectified_left)
        
        right_path = filedialog.asksaveasfilename(
            title="Save Right Rectified Image",
            defaultextension=".png",
            filetypes=[("PNG files", "*.png"), ("JPEG files", "*.jpg")]
        )
        
        if right_path:
            cv2.imwrite(right_path, self.rectifier.rectified_right)
            self.status_var.set("Rectified images saved")
    
    def _save_depth_map(self):
        """Save depth map to file."""
        if self.depth_estimator.disparity is None:
            messagebox.showwarning("Warning", "No depth map to save")
            return
        
        file_path = filedialog.asksaveasfilename(
            title="Save Depth Map",
            defaultextension=".png",
            filetypes=[("PNG files", "*.png"), ("JPEG files", "*.jpg")]
        )
        
        if file_path:
            colormap_name = self.colormap_var.get()
            colormap_map = {
                'JET': cv2.COLORMAP_JET,
                'VIRIDIS': cv2.COLORMAP_VIRIDIS,
                'MAGMA': cv2.COLORMAP_MAGMA,
                'INFERNO': cv2.COLORMAP_INFERNO,
                'PLASMA': cv2.COLORMAP_PLASMA,
                'CIVIDIS': cv2.COLORMAP_CIVIDIS
            }
            colormap = colormap_map.get(colormap_name, cv2.COLORMAP_JET)
            
            colored = self.depth_estimator.apply_colormap(self.depth_estimator.disparity, colormap)
            cv2.imwrite(file_path, colored)
            self.status_var.set("Depth map saved")
    
    def _save_calibration(self):
        """Save calibration parameters to JSON file."""
        file_path = filedialog.asksaveasfilename(
            title="Save Calibration",
            defaultextension=".json",
            filetypes=[("JSON files", "*.json")]
        )
        
        if file_path:
            calibration = {
                'left_camera': {
                    'K': self.left_param_panel.get_K().tolist(),
                    'distortion': self.left_param_panel.get_distortion().tolist()
                },
                'right_camera': {
                    'K': self.right_param_panel.get_K().tolist(),
                    'distortion': self.right_param_panel.get_distortion().tolist(),
                    'R': self.right_param_panel.get_R().tolist(),
                    'T': self.right_param_panel.get_T().tolist()
                }
            }
            
            with open(file_path, 'w') as f:
                json.dump(calibration, f, indent=2)
            
            self.status_var.set(f"Calibration saved to {os.path.basename(file_path)}")
    
    def _load_calibration(self):
        """Load calibration parameters from JSON file."""
        file_path = filedialog.askopenfilename(
            title="Load Calibration",
            filetypes=[("JSON files", "*.json")]
        )
        
        if file_path:
            try:
                with open(file_path, 'r') as f:
                    calibration = json.load(f)
                
                left_cam = calibration['left_camera']
                right_cam = calibration['right_camera']
                
                self.left_param_panel.set_K(np.array(left_cam['K']))
                self.left_param_panel.set_distortion(np.array(left_cam['distortion']))
                
                self.right_param_panel.set_K(np.array(right_cam['K']))
                self.right_param_panel.set_distortion(np.array(right_cam['distortion']))
                self.right_param_panel.set_R(np.array(right_cam['R']))
                self.right_param_panel.set_T(np.array(right_cam['T']))
                
                self._update_rectification()
                self.status_var.set(f"Calibration loaded from {os.path.basename(file_path)}")
                
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load calibration: {str(e)}")
    
    def run(self):
        """Start the application."""
        self.root.mainloop()


def main():
    """Main entry point."""
    app = StereoCalibrationGUI()
    app.run()


if __name__ == '__main__':
    main()
