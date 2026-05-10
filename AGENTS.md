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

### 3. ImagePanel `zoom_var` attribute
The `zoom_var` attribute only exists when `show_controls=True`. Always check:
```python
if hasattr(self, 'zoom_var') and self.zoom_var is not None:
    self.zoom_var.set(...)
```

### 4. StereoBM `blockSize` must be odd
Auto-corrected in `gui/main_window.py:_update_depth()`:
```python
if params['blockSize'] % 2 == 0:
    params['blockSize'] += 1
```

### 5. Intrinsic matrix validation
Focal lengths (fx, fy) must be positive. `core/rectifier.py` auto-corrects to image dimensions with warning.

### 6. Depth panel tooltip callback
The depth panel uses a `value_callback` to display disparity/depth values. The callback `_get_depth_value()` in `main_window.py` reads from `depth_estimator.disparity` and calculates depth in mm using camera parameters.

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
# Run all tests (all 31 must pass)
python tests/test_all.py           # Core: 17 tests
python tests/test_gui.py           # GUI: 6 tests (requires X11 display)
python tests/test_integration.py   # Integration: 7 tests
python tests/test_raft_integration.py  # RAFT: 1 test

# GUI tests require X11 display (use xvfb-run for headless)
xvfb-run python tests/test_gui.py
```

## Architecture

```
main.py              # Entry point
core/
  rectifier.py       # OpenCV stereoRectify
  depth.py           # StereoBM, SGBM, RAFT-Stereo depth estimation
  RAFT-Stereo/       # Git submodule (Princeton VL)
gui/
  main_window.py     # Main application
  param_panel.py     # Camera parameter inputs (K, distortion, R, T)
  image_panel.py     # Image display with zoom/pan/tooltip
scripts/
  download_raft_models.py  # Model download utility
models/              # Pretrained RAFT-Stereo models
```

## Three Algorithms

### StereoBM (Block Matching)
- **Fast**, real-time capable
- 6 parameters: numDisparities, blockSize, minDisparity, uniquenessRatio, speckleWindowSize, speckleRange
- Best for: texture-rich scenes, quick testing

### StereoSGBM (Semi-Global Block Matching)
- **Slower** (2-3x BM), better quality
- 9 parameters: adds P1, P2, preFilterCap
- Best for: smooth surfaces, edge preservation

### RAFT-Stereo (Deep Learning)
- **Requires**: PyTorch, pretrained model (~45MB)
- **Slow on CPU**, fast on GPU
- 2 parameters: valid_iters, n_downsample
- Best for: challenging scenes, state-of-the-art accuracy
- **Models**: middlebury (recommended), eth3d, sceneflow, realtime

## GUI Layout

- **Left**: Camera parameters (scrollable)
- **Center**: Rectified left/right images
- **Right**: 
  - Depth visualization panel (tooltip shows disparity/depth)
  - Algorithm selector (BM | SGBM | RAFT)
  - Algorithm-specific parameters (dynamic based on selection)
  - Visualization Controls (view mode: disparity/depth, colormap)

## Visualization Controls

- **View mode**: Toggle between "disparity" (pixels) and "depth (mm)"
  - Depth formula: `depth = (baseline × focal_length) / disparity × 1000`
- **Colormap**: JET, VIRIDIS, MAGMA, INFERNO, PLASMA, CIVIDIS

## Example Data

Use synthetic examples in `examples/`:
```bash
examples/circles_left.png
examples/circles_right.png
examples/circles_calibration.json
```

All examples have ground truth calibration. See `examples/README.md` for recommended parameters per dataset.

## Calibration File Format

```json
{
  "left_camera": {"K": [[fx,0,cx],[0,fy,cy],[0,0,1]], "distortion": [k1,k2,p1,p2,k3]},
  "right_camera": {"K": [[...]], "distortion": [...], "R": [[...]], "T": [tx,ty,tz]}
}
```

## Key Dependencies

**Core**: opencv-python>=4.10.0, numpy>=2.0.0, Pillow>=8.0.0, tkinter

**RAFT-Stereo (optional)**: torch>=1.7.0, torchvision>=0.8.1, gdown

## Known Limitations

1. Synthetic examples work best; real images need proper calibration
2. Textureless regions produce poor disparity (all algorithms)
3. GUI tests require X11 display (use `xvfb-run` for headless)
4. Depth visualization clips to 10000mm maximum
5. RAFT-Stereo: First load is slow (model initialization), GPU recommended
6. **Depth values are NOT validated** - experimental only (see README warning)

## Files to Read First

1. `core/depth.py` - Depth estimation logic, RAFT integration
2. `gui/main_window.py` - GUI logic, algorithm switching, error handling
3. `core/raft_stereo_check.py` - RAFT availability detection
4. `tests/test_raft_integration.py` - RAFT testing examples
