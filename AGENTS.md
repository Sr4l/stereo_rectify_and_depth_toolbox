# Stereo Calibration Toolbox - Agent Instructions

## Quick Start

```bash
source venv/bin/activate
python main.py
```

## Critical Gotchas (Learned from Bugs)

### 1. ImagePanel `zoom_var` attribute
When modifying `gui/image_panel.py`, the `zoom_var` attribute only exists when `show_controls=True`. Always check before accessing:
```python
if hasattr(self, 'zoom_var') and self.zoom_var is not None:
    self.zoom_var.set(...)
```

### 2. StereoBM `blockSize` must be odd
OpenCV StereoBM requires odd `blockSize` (5-255). The GUI automatically corrects even values in `gui/main_window.py:_update_depth()`:
```python
if params['blockSize'] % 2 == 0:
    params['blockSize'] += 1
```

### 3. Intrinsic matrix validation
Focal lengths (fx, fy) must be positive. `core/rectifier.py` auto-corrects invalid values to image dimensions with a warning.

## Testing

```bash
# Run all tests
python tests/test_all.py          # Core: 17 tests
python tests/test_gui.py          # GUI: 6 tests (requires display)
python tests/test_integration.py  # Integration: 7 tests
```

All 30 tests must pass.

## Example Data

Use synthetic stereo pairs in `examples/` for testing:
```bash
examples/circles_left.png
examples/circles_right.png
examples/circles_calibration.json
```

These have known ground truth and work reliably with the toolbox.

## Architecture

```
main.py              # Entry point
core/
  rectifier.py       # OpenCV stereoRectify
  depth.py           # StereoBM depth estimation
gui/
  main_window.py     # Main GUI (single window layout)
  param_panel.py     # Camera parameter inputs
  image_panel.py     # Image display with zoom/pan
```

## GUI Layout

- **Left panel**: Camera parameters (K, distortion, R, T) - scrollable
- **Center-top**: Rectified left/right images side-by-side
- **Bottom-right**: Depth map with 6 BM parameter controls (compact 3-column layout)

## BM Parameter Controls

Labels are shortened for space. Mapping:
- "Num Disp:" → numDisparities (must be multiple of 16)
- "Block:" → blockSize (auto-corrected to odd)
- "Min Disp:" → minDisparity
- "Unique:" → uniquenessRatio
- "Speckle W:" → speckleWindowSize
- "Speckle R:" → speckleRange

## Calibration Files

JSON format with ground truth:
```json
{
  "left_camera": {"K": [[...]], "distortion": [...]},
  "right_camera": {"K": [[...]], "distortion": [...], "R": [[...]], "T": [...]}
}
```

Use `examples/generate_examples.py` to create more test pairs.

## Dependencies

- opencv-python>=4.5.0
- numpy>=1.19.0
- Pillow>=8.0.0
- tkinter (standard library)

## Known Limitations

1. Synthetic examples work best; real images need proper calibration
2. Textureless regions (smooth spheres) produce poor disparity
3. GUI tests require X11 display
