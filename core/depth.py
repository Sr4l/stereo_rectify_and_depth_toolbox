import numpy as np
import cv2
from typing import Optional, Tuple, Dict, Any, List
from dataclasses import dataclass


def normalize_stereo_pair(
    left_image: np.ndarray,
    right_image: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Normalize brightness and contrast of stereo image pair.
    
    Applies a two-stage normalization:
    1. Global mean-variance matching (affine transform)
    2. CLAHE for local illumination variations
    
    Works with both grayscale and color images.
    
    Args:
        left_image: Left image (grayscale or BGR color)
        right_image: Right image (grayscale or BGR color)
    
    Returns:
        Tuple of (left_normalized, right_normalized) as uint8 grayscale
    """
    is_color = len(left_image.shape) == 3
    
    if is_color:
        left_lab = cv2.cvtColor(left_image, cv2.COLOR_BGR2LAB).astype(np.float32)
        right_lab = cv2.cvtColor(right_image, cv2.COLOR_BGR2LAB).astype(np.float32)
        left_l = left_lab[:, :, 0]
        right_l = right_lab[:, :, 0]
    else:
        left_l = left_image.astype(np.float32)
        right_l = right_image.astype(np.float32)
    
    left_mean, left_std = cv2.meanStdDev(left_l)
    right_mean, right_std = cv2.meanStdDev(right_l)
    
    left_mean = left_mean[0][0]
    left_std = left_std[0][0]
    right_mean = right_mean[0][0]
    right_std = right_std[0][0]
    
    target_mean = (left_mean + right_mean) / 2.0
    target_std = (left_std + right_std) / 2.0
    
    def match_stats(img, img_mean, img_std):
        if img_std > 0:
            normalized = (img - img_mean) / img_std * target_std + target_mean
        else:
            normalized = img - img_mean + target_mean
        normalized = np.clip(normalized, 0, 255).astype(np.uint8)
        return normalized
    
    left_matched = match_stats(left_l, left_mean, left_std)
    right_matched = match_stats(right_l, right_mean, right_std)
    
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(16, 16))
    
    left_clahe = clahe.apply(left_matched)
    right_clahe = clahe.apply(right_matched)
    
    return left_clahe, right_clahe


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


@dataclass
class StereoSGBMParams:
    """Parameters for StereoSGBM algorithm."""
    numDisparities: int = 16
    blockSize: int = 9
    minDisparity: int = 0
    uniquenessRatio: int = 10
    speckleWindowSize: int = 100
    speckleRange: int = 1
    disp12MaxDiff: int = 1
    preFilterCap: int = 31
    P1: int = 200
    P2: int = 400
    mode: int = 0


@dataclass
class RAFTStereoParams:
    """Parameters for RAFT-Stereo algorithm."""
    hidden_dims: list = None
    n_gru_layers: int = 3
    corr_levels: int = 4
    corr_radius: int = 4
    n_downsample: int = 2
    context_norm: str = "batch"
    shared_backbone: bool = False
    slow_fast_gru: bool = False
    corr_implementation: str = "reg"
    valid_iters: int = 32
    mixed_precision: bool = False


# Check PyTorch availability
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

# RAFT-Stereo path and setup
import os as _os
import sys as _sys
RAFT_STEREO_PATH = _os.path.join(_os.path.dirname(__file__), 'RAFT-Stereo')

# Set up RAFT-Stereo paths if it exists
if _os.path.exists(RAFT_STEREO_PATH):
    _raft_path = _os.path.abspath(RAFT_STEREO_PATH)
    _raft_core = _os.path.join(_raft_path, 'core')
    
    # Add to sys.path
    for _path in [_raft_path, _raft_core]:
        if _path not in _sys.path:
            _sys.path.insert(0, _path)
    
    # Extend 'core' package path
    if 'core' in _sys.modules:
        _core_module = _sys.modules['core']
        if _raft_core not in _core_module.__path__:
            _core_module.__path__.append(_raft_core)
    
    # Extend 'core.utils' package path
    _raft_utils = _os.path.join(_raft_path, 'core', 'utils')
    if 'core.utils' in _sys.modules and _os.path.exists(_raft_utils):
        _utils_module = _sys.modules['core.utils']
        if _raft_utils not in _utils_module.__path__:
            _utils_module.__path__.append(_raft_utils)


class DepthEstimator:
    """Handles depth estimation using StereoBM, StereoSGBM, or RAFT-Stereo."""
    
    def __init__(self):
        self.bm_params = StereoBMParams()
        self.sgbm_params = StereoSGBMParams()
        self.raft_params = RAFTStereoParams()
        self.algorithm = 'BM'
        self.disparity: Optional[np.ndarray] = None
        self.depth_map: Optional[np.ndarray] = None
        self.baseline: float = 1.0
        self.focal_length: float = 1.0
        self.raft_model = None
        self.raft_model_path: Optional[str] = None
    
    def set_algorithm(self, algorithm: str):
        """Set the stereo matching algorithm ('BM', 'SGBM', or 'RAFT')."""
        self.algorithm = algorithm.upper()
    
    def set_bm_params(self, **kwargs):
        """Update StereoBM parameters."""
        for key, value in kwargs.items():
            if hasattr(self.bm_params, key):
                setattr(self.bm_params, key, value)
    
    def set_sgbm_params(self, **kwargs):
        """Update StereoSGBM parameters."""
        for key, value in kwargs.items():
            if hasattr(self.sgbm_params, key):
                setattr(self.sgbm_params, key, value)
    
    def set_raft_params(self, **kwargs):
        """Update RAFT-Stereo parameters."""
        for key, value in kwargs.items():
            if hasattr(self.raft_params, key):
                setattr(self.raft_params, key, value)
    
    def set_camera_params(self, baseline: float, focal_length: float):
        """Set camera parameters for depth calculation."""
        self.baseline = baseline
        self.focal_length = focal_length
    
    def load_raft_model(self, model_path: str):
        """
        Load pretrained RAFT-Stereo model.
        
        Args:
            model_path: Path to .pth checkpoint file
        
        Returns:
            RAFT-Stereo model wrapped in DataParallel
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch not installed. Install with: pip install torch torchvision")
        
        try:
            import sys
            import os
            import importlib.util
            
            # Add RAFT-Stereo paths
            raft_path = os.path.abspath(RAFT_STEREO_PATH)
            raft_core = os.path.join(raft_path, 'core')
            
            for path in [raft_path, raft_core]:
                if path not in sys.path:
                    sys.path.insert(0, path)
            
            # Manually import core.utils.utils from RAFT-Stereo
            raft_utils_path = os.path.join(raft_core, 'utils', 'utils.py')
            if os.path.exists(raft_utils_path) and 'core.utils.utils' not in sys.modules:
                utils_spec = importlib.util.spec_from_file_location(
                    "core.utils.utils",
                    raft_utils_path
                )
                utils_module = importlib.util.module_from_spec(utils_spec)
                sys.modules['core.utils.utils'] = utils_module
                utils_spec.loader.exec_module(utils_module)
            
            from core.raft_stereo import RAFTStereo
        except ImportError as e:
            raise ImportError(
                f"RAFT-Stereo module not found. Clone from https://github.com/princeton-vl/RAFT-Stereo\nError: {e}"
            )
        
        class Args:
            hidden_dims = self.raft_params.hidden_dims or [128, 128, 128]
            corr_implementation = self.raft_params.corr_implementation
            shared_backbone = self.raft_params.shared_backbone
            corr_levels = self.raft_params.corr_levels
            corr_radius = self.raft_params.corr_radius
            n_downsample = self.raft_params.n_downsample
            context_norm = self.raft_params.context_norm
            slow_fast_gru = self.raft_params.slow_fast_gru
            n_gru_layers = self.raft_params.n_gru_layers
            mixed_precision = self.raft_params.mixed_precision
        
        print(f"Loading RAFT-Stereo model from: {model_path}")
        device_ids = [0] if torch.cuda.is_available() else []
        model = torch.nn.DataParallel(RAFTStereo(Args()), device_ids=device_ids)
        model.load_state_dict(torch.load(model_path, map_location='cpu'))
        model = model.module
        
        if torch.cuda.is_available():
            model.cuda()
        
        model.eval()
        self.raft_model = model
        self.raft_model_path = model_path
        print(f"RAFT-Stereo model loaded successfully")
    
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
            left_gray, right_gray = normalize_stereo_pair(left_rectified, right_rectified)
            
            block_size = self.bm_params.blockSize
            if block_size % 2 == 0:
                block_size += 1
            block_size = max(5, min(255, block_size))
            
            num_disparities = self.bm_params.numDisparities
            num_disparities = max(16, min(256, num_disparities))
            if num_disparities % 16 != 0:
                num_disparities = ((num_disparities // 16) + 1) * 16
            
            sbm = cv2.StereoBM_create(
                numDisparities=num_disparities,
                blockSize=block_size
            )
            
            sbm.setMinDisparity(self.bm_params.minDisparity)
            sbm.setUniquenessRatio(self.bm_params.uniquenessRatio)
            sbm.setSpeckleWindowSize(self.bm_params.speckleWindowSize)
            sbm.setSpeckleRange(self.bm_params.speckleRange)
            sbm.setDisp12MaxDiff(self.bm_params.disp12MaxDiff)
            sbm.setPreFilterCap(self.bm_params.preFilterCap)
            sbm.setTextureThreshold(int(self.bm_params.textureThreshold))
            
            self.disparity = sbm.compute(left_gray, right_gray)
            self.disparity = self.disparity.astype(np.float32) / 16.0
            
            return self.disparity
            
        except Exception as e:
            print(f"Error computing disparity (BM): {e}")
            return None
    
    def compute_disparity_sgbm(
        self, 
        left_rectified: np.ndarray, 
        right_rectified: np.ndarray
    ) -> Optional[np.ndarray]:
        """
        Compute disparity map using StereoSGBM.
        
        Args:
            left_rectified: Rectified left image (grayscale or color)
            right_rectified: Rectified right image (grayscale or color)
        
        Returns:
            Disparity map or None on failure
        """
        if left_rectified is None or right_rectified is None:
            return None
        
        try:
            left_gray, right_gray = normalize_stereo_pair(left_rectified, right_rectified)
            
            block_size = self.sgbm_params.blockSize
            if block_size % 2 == 0:
                block_size += 1
            block_size = max(5, min(255, block_size))
            
            num_disparities = self.sgbm_params.numDisparities
            num_disparities = max(16, min(256, num_disparities))
            if num_disparities % 16 != 0:
                num_disparities = ((num_disparities // 16) + 1) * 16
            
            sgbm = cv2.StereoSGBM_create(
                numDisparities=num_disparities,
                blockSize=block_size,
                P1=self.sgbm_params.P1,
                P2=self.sgbm_params.P2,
                preFilterCap=self.sgbm_params.preFilterCap,
                uniquenessRatio=self.sgbm_params.uniquenessRatio,
                speckleWindowSize=self.sgbm_params.speckleWindowSize,
                speckleRange=self.sgbm_params.speckleRange,
                disp12MaxDiff=self.sgbm_params.disp12MaxDiff,
                mode=self.sgbm_params.mode
            )
            
            sgbm.setMinDisparity(self.sgbm_params.minDisparity)
            
            self.disparity = sgbm.compute(left_gray, right_gray)
            self.disparity = self.disparity.astype(np.float32) / 16.0
            
            return self.disparity
            
        except Exception as e:
            print(f"Error computing disparity (SGBM): {e}")
            return None
    
    def compute_disparity_raft(
        self,
        left_rectified: np.ndarray,
        right_rectified: np.ndarray,
        model_path: Optional[str] = None
    ) -> Optional[np.ndarray]:
        """
        Compute disparity map using RAFT-Stereo.
        
        Args:
            left_rectified: Rectified left image (grayscale or color)
            right_rectified: Rectified right image (grayscale or color)
            model_path: Path to model checkpoint (lazy loads if None)
        
        Returns:
            Disparity map (H, W) as float32, or None on failure
        """
        if not TORCH_AVAILABLE:
            print("Error: PyTorch not installed. Cannot use RAFT-Stereo.")
            return None
        
        if left_rectified is None or right_rectified is None:
            return None
        
        from core.utils.input_padder import InputPadder
        
        if self.raft_model is None:
            if model_path is None:
                raise ValueError("RAFT-Stereo model not loaded. Provide model_path.")
            self.load_raft_model(model_path)
        
        try:
            if len(left_rectified.shape) == 2:
                left_rectified = cv2.cvtColor(left_rectified, cv2.COLOR_GRAY2RGB)
                right_rectified = cv2.cvtColor(right_rectified, cv2.COLOR_GRAY2RGB)
            
            image1 = torch.from_numpy(left_rectified).permute(2, 0, 1).float()[None]
            image2 = torch.from_numpy(right_rectified).permute(2, 0, 1).float()[None]
            
            image1 = (2 * (image1 / 255.0) - 1.0)
            image2 = (2 * (image2 / 255.0) - 1.0)
            
            if torch.cuda.is_available():
                image1 = image1.cuda()
                image2 = image2.cuda()
            
            padder = InputPadder(image1.shape, divis_by=32)
            image1, image2 = padder.pad(image1, image2)
            
            with torch.no_grad():
                _, disparity = self.raft_model(
                    image1, image2,
                    iters=self.raft_params.valid_iters,
                    test_mode=True
                )
                disparity = padder.unpad(disparity).squeeze()
            
            if torch.cuda.is_available():
                disparity = disparity.cpu()
            
            self.disparity = disparity.numpy().astype(np.float32)
            return self.disparity
            
        except Exception as e:
            print(f"Error computing disparity (RAFT-Stereo): {e}")
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
