import numpy as np
import cv2
from typing import Optional, Tuple, Dict, Any
from dataclasses import dataclass


@dataclass
class StereoBMParams:
    """Parameters for StereoBM algorithm."""
    numDisparities: int = 16
    blockSize: int = 9
    minDisparity: int = 0
    uniquenessRatio: int = 10
    speckleWindowSize: int = 100
    speckleRange: int = 1
    disp12MaxDiff: int = 1
    preFilterCap: int = 31
    textureThreshold: float = 10.0
    mode: int = 0


class DepthEstimator:
    """Handles depth estimation using StereoBM."""
    
    def __init__(self):
        self.params = StereoBMParams()
        self.disparity: Optional[np.ndarray] = None
        self.depth_map: Optional[np.ndarray] = None
        self.baseline: float = 1.0
        self.focal_length: float = 1.0
    
    def set_bm_params(self, **kwargs):
        """Update StereoBM parameters."""
        for key, value in kwargs.items():
            if hasattr(self.params, key):
                setattr(self.params, key, value)
    
    def set_camera_params(self, baseline: float, focal_length: float):
        """Set camera parameters for depth calculation."""
        self.baseline = baseline
        self.focal_length = focal_length
    
    def compute_disparity(
        self, 
        left_rectified: np.ndarray, 
        right_rectified: np.ndarray
    ) -> Optional[np.ndarray]:
        """
        Compute disparity map from rectified stereo images.
        
        Args:
            left_rectified: Rectified left image (grayscale or color)
            right_rectified: Rectified right image (grayscale or color)
        
        Returns:
            Disparity map or None on failure
        """
        if left_rectified is None or right_rectified is None:
            return None
        
        try:
            if len(left_rectified.shape) == 3:
                left_gray = cv2.cvtColor(left_rectified, cv2.COLOR_BGR2GRAY)
            else:
                left_gray = left_rectified
            
            if len(right_rectified.shape) == 3:
                right_gray = cv2.cvtColor(right_rectified, cv2.COLOR_BGR2GRAY)
            else:
                right_gray = right_rectified
            
            sbm = cv2.StereoBM_create(
                numDisparities=self.params.numDisparities,
                blockSize=self.params.blockSize
            )
            
            sbm.setMinDisparity(self.params.minDisparity)
            sbm.setUniquenessRatio(self.params.uniquenessRatio)
            sbm.setSpeckleWindowSize(self.params.speckleWindowSize)
            sbm.setSpeckleRange(self.params.speckleRange)
            sbm.setDisp12MaxDiff(self.params.disp12MaxDiff)
            sbm.setPreFilterCap(self.params.preFilterCap)
            sbm.setTextureThreshold(int(self.params.textureThreshold))
            
            self.disparity = sbm.compute(left_gray, right_gray)
            self.disparity = self.disparity.astype(np.float32) / 16.0
            
            return self.disparity
            
        except Exception as e:
            print(f"Error computing disparity: {e}")
            return None
    
    def compute_depth(
        self, 
        disparity: Optional[np.ndarray] = None,
        baseline: Optional[float] = None,
        focal_length: Optional[float] = None
    ) -> Optional[np.ndarray]:
        """
        Convert disparity to depth map.
        
        Depth = (baseline * focal_length) / disparity
        
        Args:
            disparity: Disparity map (uses cached if None)
            baseline: Camera baseline in meters (uses instance value if None)
            focal_length: Focal length in pixels (uses instance value if None)
        
        Returns:
            Depth map in meters or None on failure
        """
        disp = disparity if disparity is not None else self.disparity
        
        if disp is None:
            return None
        
        bl = baseline if baseline is not None else self.baseline
        fl = focal_length if focal_length is not None else self.focal_length
        
        try:
            with np.errstate(divide='ignore', invalid='ignore'):
                self.depth_map = (bl * fl) / disp
                self.depth_map[~np.isfinite(self.depth_map)] = 0
                self.depth_map[self.depth_map < 0] = 0
            
            return self.depth_map
            
        except Exception as e:
            print(f"Error computing depth: {e}")
            return None
    
    def apply_colormap(
        self, 
        disparity: Optional[np.ndarray] = None,
        colormap: int = cv2.COLORMAP_JET
    ) -> Optional[np.ndarray]:
        """
        Apply colormap to disparity map for visualization.
        
        Args:
            disparity: Disparity map (uses cached if None)
            colormap: OpenCV colormap constant
        
        Returns:
            Colorized disparity map or None on failure
        """
        disp = disparity if disparity is not None else self.disparity
        
        if disp is None:
            return None
        
        try:
            disp_normalized = disp.copy()
            min_val = np.min(disp_normalized)
            max_val = np.max(disp_normalized)
            
            if max_val > min_val:
                disp_normalized = ((disp_normalized - min_val) / (max_val - min_val) * 255).astype(np.uint8)
            else:
                disp_normalized = np.zeros_like(disp_normalized, dtype=np.uint8)
            
            colored = cv2.applyColorMap(disp_normalized, colormap)
            
            return colored
            
        except Exception as e:
            print(f"Error applying colormap: {e}")
            return None
    
    def get_disparity_stats(self) -> Dict[str, float]:
        """Get statistics about the current disparity map."""
        if self.disparity is None:
            return {
                'min': 0.0,
                'max': 0.0,
                'mean': 0.0,
                'std': 0.0
            }
        
        valid = self.disparity[self.disparity > 0]
        if len(valid) == 0:
            return {
                'min': 0.0,
                'max': 0.0,
                'mean': 0.0,
                'std': 0.0
            }
        
        return {
            'min': float(np.min(valid)),
            'max': float(np.max(valid)),
            'mean': float(np.mean(valid)),
            'std': float(np.std(valid))
        }
    
    def get_depth_stats(self) -> Dict[str, float]:
        """Get statistics about the current depth map."""
        if self.depth_map is None:
            return {
                'min': 0.0,
                'max': 0.0,
                'mean': 0.0,
                'std': 0.0
            }
        
        valid = self.depth_map[self.depth_map > 0]
        if len(valid) == 0:
            return {
                'min': 0.0,
                'max': 0.0,
                'mean': 0.0,
                'std': 0.0
            }
        
        return {
            'min': float(np.min(valid)),
            'max': float(np.max(valid)),
            'mean': float(np.mean(valid)),
            'std': float(np.std(valid))
        }
