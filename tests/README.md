# Test Suite Documentation

## Overview

The Stereo Camera Calibration & Depth Toolbox includes a comprehensive test suite with **32 tests** across four categories:
- Core functionality tests (17 tests)
- GUI component tests (6 tests)
- Integration tests (7 tests)
- Linting tests (2 tests - ruff check + ruff format)

## Running Tests

### Run All Tests
```bash
source venv/bin/activate

# Run individual test suites
python tests/test_all.py          # Core functionality
python tests/test_gui.py          # GUI components
python tests/test_integration.py  # Integration tests
python tests/test_linting.py      # Linting (ruff)

# Or run all at once
python tests/test_all.py && python tests/test_gui.py && python tests/test_integration.py && python tests/test_linting.py
```

## Test Categories

### 1. Core Tests (`tests/test_all.py`)

Tests the core rectification and depth estimation modules.

**StereoRectifier Tests:**
- ✓ Initialization
- ✓ Set images
- ✓ Set intrinsics (K matrix, distortion)
- ✓ Set extrinsics (R, T)
- ✓ Get image size
- ✓ Rectification
- ✓ Draw epipolar lines

**DepthEstimator Tests:**
- ✓ Initialization
- ✓ Set BM parameters
- ✓ Set camera parameters
- ✓ Compute disparity
- ✓ Compute depth
- ✓ Apply colormap
- ✓ Get statistics

**Additional Tests:**
- ✓ StereoBMParams dataclass
- ✓ All example images (sphere, circles, dots, checkerboard, indoor)
- ✓ Edge cases (None handling, invalid parameters)

### 2. GUI Tests (`tests/test_gui.py`)

Tests the Tkinter GUI components.

- ✓ Module imports
- ✓ CameraParamPanel (get/set all parameters)
- ✓ ThumbnailPanel (image display)
- ✓ ImagePanel (zoom, pan, controls)
- ✓ StereoCalibrationGUI initialization
- ✓ StereoCalibrationGUI with example images

### 3. Integration Tests (`tests/test_integration.py`)

Tests complete workflows and end-to-end functionality.

- ✓ Complete workflow (sphere example)
- ✓ All example pairs processing
- ✓ BM parameter sensitivity
- ✓ Calibration save/load
- ✓ Image format support (PNG)
- ✓ Rectification quality
- ✓ Depth value range validation

## Test Coverage

### Modules Tested
- `core/rectifier.py` - Stereo rectification
- `core/depth.py` - Depth estimation
- `gui/param_panel.py` - Parameter input panel
- `gui/image_panel.py` - Image display panels
- `gui/main_window.py` - Main application window

### Linting Tests (`tests/test_linting.py`)

Uses [ruff](https://github.com/astral-sh/ruff) for fast Python linting:

- ✓ Ruff check (linting rules)
- ✓ Ruff format (code formatting)

Note: The linter excludes `core/RAFT-Stereo/` (third-party submodule), `venv/`, `__pycache__/`, and other non-project directories.

### Example Data Tested
All 5 example stereo pairs are tested:
1. **sphere** - 3D sphere with depth variation
2. **circles** - Concentric circles pattern
3. **dots** - Random dot stereogram
4. **checkerboard** - Angled chessboard
5. **indoor** - Synthetic room scene

## Test Results Summary

```
Core Tests:        17/17 passed ✓
GUI Tests:          6/6  passed ✓
Integration Tests:  7/7  passed ✓
Linting Tests:      2/2  passed ✓ (if code passes ruff)
---------------------------------
Total:             32/32 passed ✓
```

## Known Limitations

1. **Synthetic Images**: Example images are synthetic and may produce different results than real-world stereo pairs
2. **Sphere Example**: The sphere example produces zero disparity with default BM parameters due to smooth texture. Increase blockSize for better results.
3. **GUI Tests**: Require display (X11) to run. Headless testing not supported.

## Adding New Tests

To add new tests:

1. Create test function following naming convention: `test_<feature>()`
2. Use assertions to validate behavior
3. Add to appropriate test file
4. Update this documentation

Example:
```python
def test_new_feature():
    """Test new feature description."""
    print("Testing new feature...")
    
    from core.module import NewClass
    
    obj = NewClass()
    result = obj.method()
    
    assert result is not None, "Result should not be None"
    assert result.shape == expected_shape, f"Shape mismatch"
    
    print("  ✓ New feature test passed")
    return True
```

## Troubleshooting

### Test fails with "No module named..."
Ensure virtual environment is activated:
```bash
source venv/bin/activate
```

### GUI tests fail with "no display"
Tests require X11 display. Use Xvfb for headless testing:
```bash
xvfb-run -a python tests/test_gui.py
```

### OpenCV assertion errors
Some tests intentionally trigger OpenCV errors to verify error handling. These are expected and tests still pass.

## Continuous Integration

To add to CI pipeline:
```yaml
test:
  script:
    - python tests/test_all.py
    - python tests/test_gui.py
    - python tests/test_integration.py
    - python tests/test_linting.py  # requires ruff installed
```

Add `ruff` to your project dependencies:
```bash
pip install ruff
```
