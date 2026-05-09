import tkinter as tk
from tkinter import ttk
from typing import Callable, Optional, Dict, List, Any
import numpy as np


class CameraParamPanel(ttk.LabelFrame):
    """Panel for editing camera intrinsic and extrinsic parameters."""
    
    def __init__(
        self, 
        parent, 
        title: str,
        on_change: Optional[Callable] = None,
        **kwargs
    ):
        super().__init__(parent, text=title, **kwargs)
        self.on_change = on_change
        self.entries: Dict[str, List[List[tk.Entry]]] = {}
        self.dist_entries: Dict[str, tk.Entry] = {}
        self.t_entries: List[tk.Entry] = []
        
        self._create_widgets()
    
    def _create_widgets(self):
        """Create all parameter input widgets."""
        main_frame = ttk.Frame(self)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self._create_intrinsic_section(main_frame)
        self._create_distortion_section(main_frame)
        self._create_extrinsic_section(main_frame)
    
    def _create_intrinsic_section(self, parent):
        """Create intrinsic matrix (K) input section."""
        frame = ttk.LabelFrame(parent, text="Intrinsic Matrix K")
        frame.pack(fill=tk.X, pady=5)
        
        labels = ['fx', 'fy', 'cx', 'cy']
        self.entries['K'] = []
        
        for i in range(3):
            row_entries = []
            row_frame = ttk.Frame(frame)
            row_frame.pack(fill=tk.X, pady=2)
            
            for j in range(3):
                cell_frame = ttk.Frame(row_frame)
                cell_frame.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=2)
                
                if i == j:
                    label_text = labels[i] if i < len(labels) else f'k{i+1}'
                else:
                    label_text = ''
                
                label = ttk.Label(cell_frame, text=label_text, width=4)
                label.pack(side=tk.LEFT)
                
                entry = ttk.Entry(cell_frame, width=10)
                entry.pack(side=tk.LEFT, fill=tk.X, expand=True)
                entry.bind('<FocusOut>', lambda e, cb=self.on_change: self._on_entry_change(cb))
                entry.bind('<Return>', lambda e, cb=self.on_change: self._on_entry_change(cb))
                
                row_entries.append(entry)
            
            self.entries['K'].append(row_entries)
        
        self._set_default_intrinsics()
    
    def _create_distortion_section(self, parent):
        """Create distortion coefficients input section."""
        frame = ttk.LabelFrame(parent, text="Distortion Coefficients")
        frame.pack(fill=tk.X, pady=5)
        
        dist_frame = ttk.Frame(frame)
        dist_frame.pack(fill=tk.X, pady=5)
        
        dist_labels = ['k1', 'k2', 'p1', 'p2', 'k3']
        self.dist_entries = {}
        
        for i, label in enumerate(dist_labels):
            cell_frame = ttk.Frame(dist_frame)
            cell_frame.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=2)
            
            lbl = ttk.Label(cell_frame, text=label, width=4)
            lbl.pack(side=tk.LEFT)
            
            entry = ttk.Entry(cell_frame, width=10)
            entry.pack(side=tk.LEFT, fill=tk.X, expand=True)
            entry.bind('<FocusOut>', lambda e, cb=self.on_change: self._on_entry_change(cb))
            entry.bind('<Return>', lambda e, cb=self.on_change: self._on_entry_change(cb))
            
            self.dist_entries[label] = entry
        
        self._set_default_distortion()
    
    def _create_extrinsic_section(self, parent):
        """Create rotation matrix and translation vector section."""
        extrinsic_frame = ttk.LabelFrame(parent, text="Extrinsic Parameters")
        extrinsic_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        self._create_rotation_matrix(extrinsic_frame)
        self._create_translation_vector(extrinsic_frame)
    
    def _create_rotation_matrix(self, parent):
        """Create rotation matrix (R) input section."""
        frame = ttk.LabelFrame(parent, text="Rotation Matrix R")
        frame.pack(fill=tk.X, pady=5)
        
        self.entries['R'] = []
        
        for i in range(3):
            row_entries = []
            row_frame = ttk.Frame(frame)
            row_frame.pack(fill=tk.X, pady=2)
            
            for j in range(3):
                cell_frame = ttk.Frame(row_frame)
                cell_frame.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=2)
                
                entry = ttk.Entry(cell_frame, width=10)
                entry.pack(side=tk.LEFT, fill=tk.X, expand=True)
                entry.bind('<FocusOut>', lambda e, cb=self.on_change: self._on_entry_change(cb))
                entry.bind('<Return>', lambda e, cb=self.on_change: self._on_entry_change(cb))
                
                row_entries.append(entry)
            
            self.entries['R'].append(row_entries)
        
        self._set_default_rotation()
    
    def _create_translation_vector(self, parent):
        """Create translation vector (T) input section."""
        frame = ttk.LabelFrame(parent, text="Translation Vector T")
        frame.pack(fill=tk.X, pady=5)
        
        vec_frame = ttk.Frame(frame)
        vec_frame.pack(fill=tk.X, pady=5)
        
        t_labels = ['Tx', 'Ty', 'Tz']
        self.t_entries = []
        
        for i, label in enumerate(t_labels):
            cell_frame = ttk.Frame(vec_frame)
            cell_frame.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=2)
            
            lbl = ttk.Label(cell_frame, text=label, width=4)
            lbl.pack(side=tk.LEFT)
            
            entry = ttk.Entry(cell_frame, width=10)
            entry.pack(side=tk.LEFT, fill=tk.X, expand=True)
            entry.bind('<FocusOut>', lambda e, cb=self.on_change: self._on_entry_change(cb))
            entry.bind('<Return>', lambda e, cb=self.on_change: self._on_entry_change(cb))
            
            self.t_entries.append(entry)
        
        self._set_default_translation()
    
    def _set_default_intrinsics(self):
        """Set default identity-like intrinsic values."""
        defaults = [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0]
        ]
        for i in range(3):
            for j in range(3):
                self.entries['K'][i][j].delete(0, tk.END)
                self.entries['K'][i][j].insert(0, str(defaults[i][j]))
    
    def _set_default_distortion(self):
        """Set default zero distortion values."""
        for key in self.dist_entries:
            self.dist_entries[key].delete(0, tk.END)
            self.dist_entries[key].insert(0, '0.0')
    
    def _set_default_rotation(self):
        """Set default identity rotation matrix."""
        for i in range(3):
            for j in range(3):
                self.entries['R'][i][j].delete(0, tk.END)
                val = 1.0 if i == j else 0.0
                self.entries['R'][i][j].insert(0, str(val))
    
    def _set_default_translation(self):
        """Set default zero translation vector."""
        for entry in self.t_entries:
            entry.delete(0, tk.END)
            entry.insert(0, '0.0')
    
    def _on_entry_change(self, callback: Optional[Callable]):
        """Handle entry value changes."""
        if callback:
            callback()
    
    def get_K(self) -> np.ndarray:
        """Get intrinsic matrix as numpy array."""
        K = np.eye(3, dtype=np.float64)
        try:
            for i in range(3):
                for j in range(3):
                    val = self.entries['K'][i][j].get().strip()
                    if val:
                        K[i, j] = float(val)
        except ValueError:
            pass
        return K
    
    def get_distortion(self) -> np.ndarray:
        """Get distortion coefficients as numpy array."""
        dist = np.zeros(5, dtype=np.float64)
        keys = ['k1', 'k2', 'p1', 'p2', 'k3']
        try:
            for i, key in enumerate(keys):
                val = self.dist_entries[key].get().strip()
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
                    val = self.entries['R'][i][j].get().strip()
                    if val:
                        R[i, j] = float(val)
        except ValueError:
            pass
        return R
    
    def get_T(self) -> np.ndarray:
        """Get translation vector as numpy array."""
        T = np.zeros(3, dtype=np.float64)
        try:
            for i, entry in enumerate(self.t_entries):
                val = entry.get().strip()
                if val:
                    T[i] = float(val)
        except ValueError:
            pass
        return T
    
    def set_K(self, K: np.ndarray):
        """Set intrinsic matrix values."""
        for i in range(3):
            for j in range(3):
                self.entries['K'][i][j].delete(0, tk.END)
                self.entries['K'][i][j].insert(0, f'{K[i, j]:.6f}')
    
    def set_distortion(self, dist: np.ndarray):
        """Set distortion coefficients."""
        keys = ['k1', 'k2', 'p1', 'p2', 'k3']
        for i, key in enumerate(keys):
            self.dist_entries[key].delete(0, tk.END)
            if i < len(dist):
                self.dist_entries[key].insert(0, f'{dist[i]:.6f}')
    
    def set_R(self, R: np.ndarray):
        """Set rotation matrix values."""
        for i in range(3):
            for j in range(3):
                self.entries['R'][i][j].delete(0, tk.END)
                self.entries['R'][i][j].insert(0, f'{R[i, j]:.6f}')
    
    def set_T(self, T: np.ndarray):
        """Set translation vector values."""
        for i, entry in enumerate(self.t_entries):
            entry.delete(0, tk.END)
            if i < len(T):
                entry.insert(0, f'{T[i]:.6f}')
    
    def reset_to_identity(self):
        """Reset all parameters to default values."""
        self._set_default_intrinsics()
        self._set_default_distortion()
        self._set_default_rotation()
        self._set_default_translation()
        if self.on_change:
            self.on_change()
