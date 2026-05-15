# Stereo Rectification & Depth Toolbox - Agent Instructions

## Quick Start

```bash
source venv/bin/activate
python main.py
```

## Critical Gotchas

### 1. RAFT-Stereo disparity sign (FIXED)
**Bug fixed in `core/depth.py:440`**: RAFT outputs optical flow (negative values), must be negated to get positive disparity:
```python
self.disparity = (-disparity).numpy().astype(np.float32)  # NOT disparity.numpy()
```

### 2. RAFT-Stereo `slow_fast_gru` parameter (FIXED)
**Bug fixed in `core/depth.py:109`**: Must be `True` for pretrained models:
```python
slow_fast_gru: bool = True  # NOT False
```

### 3. StereoBM `blockSize` must be odd
Auto-corrected in `core/depth.py:294` and `core/depth.py:347`:
```python
block_size = self.bm_params.blockSize
if block_size % 2 == 0:
    block_size += 1
block_size = max(5, min(255, block_size))
```

### 4. Intrinsic matrix validation
Focal lengths (fx, fy) must be positive. `core/rectifier.py` auto-corrects to image dimensions with warning.

### 5. Camera calibration units (CRITICAL)
The translation vector `T` from calibration must be in **meters**.
The code uses meters consistently throughout for all depth calculations and visualization.

**Depth formula**: `depth (meters) = (baseline_meters * focal_length_pixels) / disparity_pixels`

**Baseline**: `np.linalg.norm(T)` - magnitude of translation vector (must be in meters)
**Focal length**: `K[0, 0]` (fx) - horizontal focal length in pixels

**If your calibration used different units** (e.g., mm from checkerboard calibration), 
you must convert T to meters before loading into the GUI:
- mm → m: divide by 1000
- cm → m: divide by 100

### 6. Depth calculation logic
```python
# core/depth.py:447-478
# Depth (meters) = (baseline_meters * focal_length_pixels) / disparity_pixels
self.depth_map = (bl * fl) / disp
self.depth_map[~np.isfinite(self.depth_map)] = 0
self.depth_map[self.depth_map < 0] = 0
```

### 7. StereoBM/SGBM disparity normalization
Both StereoBM and StereoSGBM output 16-bit fixed-point disparity maps that must be divided by 16.0:
```python
self.disparity = self.disparity.astype(np.float32) / 16.0  # NOT self.disparity directly
```

### 8. Raft model lazy loading
RAFT model can be lazily loaded via `model_path` parameter in `compute_disparity_raft()`. If not provided, raises `ValueError`. The `load_raft_model()` method can be called separately to pre-load.

### 9. Stereo image normalization
`normalize_stereo_pair()` in `core/depth.py` applies two-stage normalization before disparity computation:
1. Global mean-variance matching (affine transform on LAB L-channel)
2. CLAHE for local illumination variations

Works with both grayscale and BGR color images.

## RAFT-Stereo Submodule

**CRITICAL**: RAFT-Stereo is a git submodule. Check if initialized:

```bash
# Check if submodule exists and is initialized
ls core/RAFT-Stereo/core/raft_stereo.py

# If missing, initialize:
git submodule update --init --recursive
```

The code detects missing submodule and shows user-friendly error with git command when RAFT is selected.

## Testing

```bash
# Run all tests (all tests must pass)
python tests/test_all.py           # Core tests
python tests/test_gui.py           # GUI tests (requires X11 display)
python tests/test_integration.py   # Integration tests
python tests/test_raft_integration.py  # RAFT tests
python tests/test_linting.py       # Linting checks

# GUI tests require X11 display (use xvfb-run for headless)
xvfb-run python tests/test_gui.py
```

## Architecture

```
main.py              # Entry point ( launches PySide6 GUI )
core/
  rectifier.py       # OpenCV stereoRectify
  depth.py           # StereoBM, SGBM, RAFT-Stereo depth estimation + normalize_stereo_pair()
  raft_stereo_check.py  # RAFT availability detection
  RAFT-Stereo/       # Git submodule (Princeton VL)
core/utils/
  input_padder.py    # Input padder for RAFT (divis_by=32 padding)
  raft_utils.py      # RAFT utility helpers
gui/
  qt_main_window.py  # Main PySide6 application window (StereoCalibrationGUI)
  qt_param_panel.py  # Camera parameter input panel (K, distortion, R, T)
  qt_image_panel.py  # Image display with zoom/pan/tooltip (QGraphicsView)
  theme.py           # Dark/light theme support
  export_dialog.py   # Export dialog for depth/disparity data
scripts/
  download_raft_models.py  # Model download utility
models/              # Pretrained RAFT-Stereo models
```

## Three Algorithms

### StereoBM (Block Matching)
- **Fast**, real-time capable
- 6 UI parameters (additional params in StereoBMParams: disp12MaxDiff, preFilterCap, textureThreshold, mode)
- Best for: texture-rich scenes, quick testing

### StereoSGBM (Semi-Global Block Matching)
- **Slower** (2-3x BM), better quality
- 9 UI parameters (adds P1, P2, preFilterCap)
- Best for: smooth surfaces, edge preservation

### RAFT-Stereo (Deep Learning)
- **Requires**: PyTorch, pretrained model (~45MB)
- **Slow on CPU**, fast on GPU
- 2 UI parameters: valid_iters, n_downsample
- Best for: challenging scenes, state-of-the-art accuracy
- **Models**: middlebury (recommended), eth3d, sceneflow, realtime

## GUI Layout (PySide6/Qt6)

```
┌─────────────────────────────────────────────────────────────────────┐
│  [Calibration]         │  [Camera Images]    │  [Depth Map]        │
│  Save/Load Calib.      │  ┌──────────┬──────┐ │  [Disparity]        │
│                          │  │ Left     │Right │ │                     │
│  [Left Camera Params]   │  │ Camera   │Cmera │ │  Algorithm Selector │
│  - fx, fy, cx, cy       │  └──────────┴──────┘ │  (BM | SGBM | RAFT) │
│  - Distortion (k1-k3)   │                      │                     │
│  - Rotation R (3x3)     │  [Synchronized Zoom │ │  BM/SGBM Parameters │
│  - Translation T (3x1)  │   Control Slider]   │  (dynamic)          │
│                         │                      │                     │
│  [Right Camera Params]  │  [Rectified Views]  │  Visualization Ctrl  │
│  - Same structure       │  ┌──────────┬──────┐ │  - View mode        │
│                         │  │ Left     │Right │ │  - Colormap         │
│  [Epipolar/View Options]│  │Rectfied  │Rectf │ │  - Range min/max    │
│  - Show Epipolar Lines  │  └──────────┴──────┘ │  - Auto range       │
│  - View: rectified/gray │                     │                     │
└─────────────────────────────────────────────────────────────────────┘
```

- **Left Panel**: Calibration buttons, scrollable camera parameters (intrinsics, distortion, extrinsics)
- **Center Panel**: 
  - Camera images (left/right) with load buttons
  - Synchronized zoom control (slider + Fit/1:1 buttons)
  - Rectified views with epipolar line toggle and view mode selector
  - Save rectified images button
- **Right Panel**:
  - Depth/disparity visualization panel (with per-image export)
  - Algorithm selector (BM | SGBM | RAFT)
  - Algorithm-specific parameter sliders (dynamic based on selection)
  - Visualization controls (view mode, colormap, min/max range with auto checkbox)
  - RAFT model path, browse, and download buttons

## Key UI Features

### Synchronized Zoom Control
All four image panels (camera left/right, rectified left/right) share synchronized zoom via a slider. Zoom values propagate across all panels with recursive-update prevention.

### Colormap Options
JET, VIRIDIS, MAGMA, INFERNO, PLASMA, CIVIDIS

### Theme Support
Dark/light theme toggle (`Ctrl+T`). Theme preference is persisted via QSettings.

### Context Menu
Right-click on any image panel for: Save Image, Reset View, Copy to Clipboard

### Keyboard Shortcuts
| Shortcut | Action |
|----------|--------|
| `Ctrl+O` | Load left image |
| `Ctrl+R` | Load right image |
| `Ctrl+S` | Save rectified images |
| `Ctrl+L` | Load calibration |
| `Ctrl+Shift+S` | Save calibration |
| `Ctrl+T` | Toggle dark/light theme |
| `F5` | Refresh rectification |

## Visualization Controls

- **View mode**: Toggle between "disparity" (pixels) and "depth (m)"
  - Depth formula: `depth (m) = (baseline × focal_length) / disparity`
- **Colormap**: JET, VIRIDIS, MAGMA, INFERNO, PLASMA, CIVIDIS
- **Range control**: Min/Max values for colormap scaling with Auto checkbox for automatic range detection

## Example Data

Use synthetic examples in `examples/`:
```bash
examples/tsukuba_lowres_left.png
examples/tsukuba_lowres_right.png
examples/tsukuba_lowres_calibration.json

examples/tsukuba_highres_left.png
examples/tsukuba_highres_right.png
examples/tsukuba_highres_calibration.json
```

All examples use the Tsukuba Stereo Dataset from CVLab, University of Tsukuba.

## Calibration File Format

```json
{
  "left_camera": {"K": [[fx,0,cx],[0,fy,cy],[0,0,1]], "distortion": [k1,k2,p1,p2,k3]},
  "right_camera": {"K": [[...]], "distortion": [...], "R": [[...]], "T": [tx,ty,tz]}
}
```

## Key Dependencies

**Core**: opencv-python>=4.10.0, numpy>=2.0.0, Pillow>=8.0.0

**GUI**: PySide6 (Qt6 for Python)

**RAFT-Stereo (optional)**: torch>=1.7.0, torchvision>=0.8.1, gdown, scipy

## Known Limitations

1. Synthetic examples work best; real images need proper calibration
2. Textureless regions produce poor disparity (all algorithms)
3. GUI tests require X11 display (use `xvfb-run` for headless)
4. Depth visualization clips to maximum values based on range settings
5. RAFT-Stereo: First load is slow (model initialization), GPU recommended
6. **Depth values are NOT validated** - experimental only (see README warning)
7. StereoBM and SGBM both use OpenCV's 16-fixed point format (divide by 16.0)