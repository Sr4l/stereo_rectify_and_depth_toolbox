# Stereo Calibration Toolbox - Agent Instructions

## Quick Start

```bash
source venv/bin/activate
python main.py
```

## Critical Gotchas

### 1. ImagePanel `zoom_var` attribute
The `zoom_var` attribute only exists when `show_controls=True`. Always check:
```python
if hasattr(self, 'zoom_var') and self.zoom_var is not None:
    self.zoom_var.set(...)
```

### 2. StereoBM `blockSize` must be odd
Auto-corrected in `gui/main_window.py:_update_depth()`:
```python
if params['blockSize'] % 2 == 0:
    params['blockSize'] += 1
```

### 3. Intrinsic matrix validation
Focal lengths (fx, fy) must be positive. `core/rectifier.py` auto-corrects to image dimensions with warning.

### 4. Depth panel tooltip callback
The depth panel uses a `value_callback` to display disparity/depth values. The callback `_get_depth_value()` in `main_window.py` reads from `depth_estimator.disparity` and calculates depth in mm using camera parameters.

## Testing

```bash
# Run all tests (all 30 must pass)
python tests/test_all.py          # Core: 17 tests
python tests/test_gui.py          # GUI: 6 tests (requires X11 display)
python tests/test_integration.py  # Integration: 7 tests
```

## Architecture

```
main.py              # Entry point
core/
  rectifier.py       # OpenCV stereoRectify
  depth.py           # StereoBM depth estimation
gui/
  main_window.py     # Main application
  param_panel.py     # Camera parameter inputs (K, distortion, R, T)
  image_panel.py     # Image display with zoom/pan/tooltip
```

## GUI Layout

- **Left**: Camera parameters (scrollable)
- **Center**: Rectified left/right images
- **Right**: 
  - Depth visualization panel (with tooltip showing disparity/depth)
  - StereoBM Parameters (6 sliders in 3 rows × 2 cols)
  - Visualization Controls (2 rows: buttons | view + colormap)

## BM Parameter Controls

- "Num Disp:" → numDisparities (multiple of 16)
- "Block:" → blockSize (auto-corrected to odd, 5-255)
- "Min Disp:" → minDisparity
- "Unique:" → uniquenessRatio
- "Speckle W:" → speckleWindowSize
- "Speckle R:" → speckleRange

## Visualization Controls

- **View mode**: Toggle between "disparity" and "depth (mm)"
  - Disparity: shows raw pixel shift
  - Depth (mm): calculates `depth = (baseline × focal_length) / disparity × 1000`
- **Colormap**: JET, VIRIDIS, MAGMA, INFERNO, PLASMA, CIVIDIS

## Example Data

Use synthetic examples in `examples/`:
```bash
examples/circles_left.png
examples/circles_right.png
examples/circles_calibration.json
```

All examples have ground truth calibration. See `examples/README.md` for recommended BM parameters per dataset.

## Calibration File Format

```json
{
  "left_camera": {"K": [[...]], "distortion": [...]},
  "right_camera": {"K": [[...]], "distortion": [...], "R": [[...]], "T": [...]}
}
```

## Dependencies

- opencv-python>=4.5.0
- numpy>=1.19.0
- Pillow>=8.0.0
- tkinter (standard library)

## Known Limitations

1. Synthetic examples work best; real images need proper calibration
2. Textureless regions produce poor disparity
3. GUI tests require X11 display (use `xvfb-run` for headless)
4. Depth visualization clips to 10000mm maximum
