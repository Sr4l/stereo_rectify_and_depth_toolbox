import numpy as np
import cv2
from typing import Optional, Tuple


class StereoRectifier:
    """Handles stereo rectification using OpenCV."""
    
    def __init__(self):
        self.left_image: Optional[np.ndarray] = None
        self.right_image: Optional[np.ndarray] = None
        self.left_K = np.eye(3, dtype=np.float64)
        self.left_dist = np.zeros(5, dtype=np.float64)
        self.right_K = np.eye(3, dtype=np.float64)
        self.right_dist = np.zeros(5, dtype=np.float64)
        self.R = np.eye(3, dtype=np.float64)
        self.T = np.zeros(3, dtype=np.float64)
        
        self.left_map1 = None
        self.left_map2 = None
        self.right_map1 = None
        self.right_map2 = None
        self.rectified_left: Optional[np.ndarray] = None
        self.rectified_right: Optional[np.ndarray] = None
    
    def set_left_image(self, image: np.ndarray):
        """Set the left camera image."""
        self.left_image = image
        self._invalidate_maps()
    
    def set_right_image(self, image: np.ndarray):
        """Set the right camera image."""
        self.right_image = image
        self._invalidate_maps()
    
    def set_left_intrinsics(self, K: np.ndarray, dist: np.ndarray):
        """Set left camera intrinsics and distortion."""
        self.left_K = K.astype(np.float64)
        self.left_dist = dist.astype(np.float64)
        self._invalidate_maps()
    
    def set_right_intrinsics(self, K: np.ndarray, dist: np.ndarray):
        """Set right camera intrinsics and distortion."""
        self.right_K = K.astype(np.float64)
        self.right_dist = dist.astype(np.float64)
        self._invalidate_maps()
    
    def set_extrinsics(self, R: np.ndarray, T: np.ndarray):
        """Set rotation matrix and translation vector between cameras."""
        self.R = R.astype(np.float64)
        self.T = T.astype(np.float64)
        self._invalidate_maps()
    
    def _invalidate_maps(self):
        """Invalidate cached rectification maps."""
        self.left_map1 = None
        self.left_map2 = None
        self.right_map1 = None
        self.right_map2 = None
        self.rectified_left = None
        self.rectified_right = None
    
    def _compute_rectification_maps(self) -> bool:
        """Compute rectification maps if images are loaded."""
        if self.left_image is None or self.right_image is None:
            return False
        
        h, w = self.left_image.shape[:2]
        
        try:
            R1, R2, P1, P2, Q, validPixROI1, validPixROI2 = cv2.stereoRectify(
                self.left_K,
                self.left_dist,
                self.right_K,
                self.right_dist,
                (w, h),
                self.R,
                self.T,
                flags=cv2.CALIB_ZERO_DISPARITY,
                alpha=1.0
            )
            
            self.left_map1, self.left_map2 = cv2.initUndistortRectifyMap(
                self.left_K,
                self.left_dist,
                R1,
                P1,
                (w, h),
                cv2.CV_32FC1
            )
            
            self.right_map1, self.right_map2 = cv2.initUndistortRectifyMap(
                self.right_K,
                self.right_dist,
                R2,
                P2,
                (w, h),
                cv2.CV_32FC1
            )
            
            return True
        except Exception as e:
            print(f"Error computing rectification maps: {e}")
            return False
    
    def rectify(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Perform stereo rectification.
        
        Returns:
            Tuple of (rectified_left, rectified_right) images or (None, None) on failure
        """
        if self.left_image is None or self.right_image is None:
            return None, None
        
        if self.left_map1 is None:
            if not self._compute_rectification_maps():
                return None, None
        
        try:
            self.rectified_left = cv2.remap(
                self.left_image,
                self.left_map1,
                self.left_map2,
                cv2.INTER_LINEAR
            )
            
            self.rectified_right = cv2.remap(
                self.right_image,
                self.right_map1,
                self.right_map2,
                cv2.INTER_LINEAR
            )
            
            return self.rectified_left, self.rectified_right
        except Exception as e:
            print(f"Error during rectification: {e}")
            return None, None
    
    def get_image_size(self) -> Optional[Tuple[int, int]]:
        """Get image dimensions (width, height)."""
        if self.left_image is not None:
            h, w = self.left_image.shape[:2]
            return w, h
        elif self.right_image is not None:
            h, w = self.right_image.shape[:2]
            return w, h
        return None
    
    def draw_epipolar_lines(self, image: np.ndarray, num_lines: int = 16) -> np.ndarray:
        """Draw horizontal epipolar lines on rectified image for verification."""
        if image is None:
            return image
        
        result = image.copy()
        h, w = result.shape[:2]
        
        if len(result.shape) == 2:
            result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)
        
        step = h // (num_lines + 1)
        for i in range(1, num_lines + 1):
            y = i * step
            cv2.line(result, (0, y), (w, y), (0, 255, 0), 1)
        
        return result
